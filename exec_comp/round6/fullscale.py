"""Full-scale offline DAG, attaches to existing R6-v2 collectors. No robot interfaces."""
import argparse,fcntl,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
from exec_comp.common import ROOT,PROJECT,save_json,digest,log
from exec_comp.round6.protocol import guard_training

PYTHON=sys.executable


def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


class Pipeline:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.state_path=self.root/'pipeline_state.json'
        self.state=dict(status='running',phase='attach_base_collection',started_wall_s=time.time(),real_robot_run=False,real_records_read=False,final_direction_check_consumed=False,
            recipe=dict(train_transitions=5_000_000,validation_transitions=500_000,a1_updates=10000,a1_batch=1024,
                body_updates=12000,body_batch=512,body_width=1024,body_curriculum='first3000:H8; then9000:H17',
                b1_pretrain=dict(horizon=17,updates=1500,batch=64),online_cycles=2,
                each_cycle=dict(train=1_000_000,validation=100_000,body_updates=6000,b1_updates=1500,b1_horizon=32,b1_batch=64),primary_metric='right wrist'),
            code_sha256={p.name:digest(p) for p in (ROOT/'round6').glob('*.py')},completed_stages=[])
        self.persist()
    def persist(self):self.state['updated_wall_s']=time.time();save_json(self.state_path,self.state)
    def phase(self,name):self.state['phase']=name;self.persist();log('r6_fullscale_phase',phase=name)
    def wait_gpu(self,gpu):
        while True:
            guard_training()
            row=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            mem,util=map(int,row.strip().split(','))
            if mem<=128 and util<=5:return
            self.state['waiting_for_gpu']=gpu;self.persist();time.sleep(15)
    def start(self,label,gpu,argv):
        self.wait_gpu(gpu);path=self.root/(label+'.log')
        if path.exists():raise FileExistsError('Refusing duplicate stage log: '+str(path))
        command=[PYTHON,'-u','-m',*argv]
        with path.open('x') as stream:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,cwd=PROJECT,start_new_session=True)
        job=dict(label=label,pid=p.pid,gpu=gpu,command=command,log=str(path),started_wall_s=time.time())
        self.state.setdefault('jobs',[]).append(job);self.persist();return p,job
    def wait(self,jobs):
        pending=list(jobs)
        while pending:
            guard_training();remaining=[]
            for p,job in pending:
                rc=p.poll()
                if rc is None:remaining.append((p,job));continue
                job.update(returncode=rc,ended_wall_s=time.time())
                if rc:raise RuntimeError(f"Stage {job['label']} exited{rc}; see {job['log']}")
                self.state['completed_stages'].append(job['label']);self.persist()
            pending=remaining
            if pending:time.sleep(15)
    def dataset_counts(self,folder):
        out={}
        for split in ('train','validation'):
            files=list(Path(folder).glob(split+'_worker*_manifest.json'));receipts=[json.loads(p.read_text()) for p in files]
            if any(r['settings'].get('schema',1)<2 for r in receipts):raise ValueError('Prototype data mixed into full-scale run')
            out[split]=sum(r['transitions'] for r in receipts)
        return out
    def attach(self):
        receipt=json.loads((self.root/'collect_jobs.json').read_text());data=self.root/'data/base'
        while True:
            guard_training();missing=[]
            for job in receipt:
                manifest=data/f"{job['split']}_worker{job['gpu']:02d}_manifest.json"
                if not manifest.exists():
                    if not live(job['pid']):raise RuntimeError(f"Base collector{job['gpu']} stopped before completion")
                    missing.append(job['gpu'])
            self.state['base_collection']=dict(missing_workers=missing,completed_counts=self.dataset_counts(data));self.persist()
            if not missing:break
            time.sleep(30)
        count=self.dataset_counts(data)
        if count['train']<5_000_000 or count['validation']<500_000:raise ValueError('Base collection below authorized full-scale target')
        self.state['completed_stages'].append('base_collection_5M');self.persist()
    def train(self,label,kind,gpu,out,body=None,initialize=None,extra=(),updates=None,horizon=17,batch=None,curriculum=0,velocity_noise=0.):
        out=Path(out)
        argv=['exec_comp.round6.train','--kind',kind,'--data',str(self.root/'data/base'),'--output',str(out),'--device',f'cuda:{gpu}',
            '--updates',str(updates),'--batch',str(batch),'--horizon',str(horizon),'--width','1024','--eval-interval','500' if kind!='b1' else '250',
            '--body-curriculum-updates',str(curriculum),'--velocity-input-noise',str(velocity_noise),'--residual-smooth-weight','.1',
            '--min-train','5000000','--min-validation','500000','--primary-wrist','right']
        if body:argv+=['--body',str(body)]
        if initialize:argv+=['--initialize',str(initialize)]
        for folder in extra:argv+=['--extra-data',str(folder)]
        if extra:argv+=['--balance-roots']
        return self.start(label,gpu,argv)
    def world_check(self,folder,label):
        job=self.start(label,2,['exec_comp.round6.validate_world','--data',str(self.root/'data/base'),'--checkpoint',str(Path(folder)/'body_best.pt'),'--device','cuda:2'])
        self.wait([job])
    def online_collect(self,index,checkpoint):
        folder=self.root/f'data/online_{index}';jobs=[]
        for gpu in range(8):
            split='validation' if gpu==7 else 'train';count=100000 if gpu==7 else math.ceil(1_000_000/7/8)*8
            jobs.append(self.start(f'collect_online{index}_{split}_{gpu}',gpu,['exec_comp.round6.collect','--split',split,'--worker',str(index*10+gpu),'--envs','8','--transitions',str(count),'--device',f'cuda:{gpu}','--output',str(folder),'--controller','b1','--checkpoint',str(checkpoint)]))
        self.wait(jobs);count=self.dataset_counts(folder)
        if count['train']<1_000_000 or count['validation']<100000:raise ValueError('Online loop incomplete')
        self.state[f'online_{index}_data']=count;self.persist();return folder
    def evaluate(self,a1,b1):
        pending=[]
        for population in ('library','brain'):
            shards=4 if population=='library' else 1
            for randomized in (False,True):
                for method,checkpoint in (('raw',None),('a1',a1),('b1',b1)):
                    for shard in range(shards):
                        label=f'eval_{population}_{method}_{"dr" if randomized else "nominal"}_shard{shard}'
                        pending.append((label,population,method,checkpoint,randomized,shard,shards))
        running=[];free=list(range(8))
        while pending or running:
            while free and pending:
                gpu=free.pop(0);label,pop,method,checkpoint,dr,shard,shards=pending.pop(0)
                argv=['exec_comp.round6.evaluate','--method',method,'--population',pop,'--clips','300' if pop=='library' else '10','--steps','100000','--batch','4','--device',f'cuda:{gpu}','--shard-index',str(shard),'--shards',str(shards)]
                if checkpoint:argv+=['--checkpoint',str(checkpoint)]
                if dr:argv+=['--randomized']
                running.append(self.start(label,gpu,argv))
            new=[]
            for p,job in running:
                rc=p.poll()
                if rc is None:new.append((p,job));continue
                job.update(returncode=rc,ended_wall_s=time.time())
                if rc:raise RuntimeError(f"Evaluation failed: {job['label']}")
                self.state['completed_stages'].append(job['label']);free.append(job['gpu']);self.persist()
            running=new
            if pending or running:time.sleep(15)
    def run(self):
        self.attach();base=self.root/'models/base'
        self.phase('train_A1_10000_and_WM_12000')
        jobs=[self.train('train_a1_full','a1',0,base,updates=10000,batch=1024),
            self.train('train_body_full','body',1,base,updates=12000,batch=512,horizon=17,curriculum=3000,velocity_noise=.1)]
        self.wait(jobs);self.world_check(base,'world_validation_base_H17_H32')
        self.phase('B1_H17_pretrain');pre=self.root/'models/b1_pretrain'
        self.wait([self.train('train_b1_H17','b1',0,pre,body=base/'body_best.pt',updates=1500,horizon=17,batch=64)])
        g=pre/'b1_best.pt';body=base/'body_best.pt';extra=[]
        for index in (1,2):
            self.phase(f'online_cycle_{index}_MuJoCo_collection');extra.append(self.online_collect(index,g))
            folder=self.root/f'models/online_{index}';self.phase(f'online_cycle_{index}_WM_refine')
            self.wait([self.train(f'train_body_online{index}','body',0,folder,initialize=body,extra=extra,updates=6000,horizon=17,batch=512)])
            self.world_check(folder,f'world_validation_online{index}_H17_H32');body=folder/'body_best.pt'
            self.phase(f'online_cycle_{index}_B1_H32')
            self.wait([self.train(f'train_b1_online{index}_H32','b1',0,folder,body=body,initialize=g,extra=extra,updates=1500,horizon=32,batch=64)])
            g=folder/'b1_best.pt'
        self.phase('paired_full_MuJoCo_evaluation_300_library_10_brain_nominal_and_DR')
        self.evaluate(base/'a1_best.pt',g)
        self.state.update(status='simulation_complete',phase='review_right_wrist_before_once_only_real_record_check',models=dict(a1=str(base/'a1_best.pt'),b1=str(g),body=str(body)),finished_wall_s=time.time())
        self.persist();log('r6_fullscale_simulation_complete',state=str(self.state_path))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pipe=Pipeline(root)
        try:pipe.run()
        except BaseException as exc:
            pipe.state.update(status='failed',error=repr(exc));pipe.persist();traceback.print_exc();raise


if __name__=='__main__':main()
