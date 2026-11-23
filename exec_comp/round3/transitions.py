"""Episode-safe body-model replay written by the real-switcher Isaac loop."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from exec_comp.common import save_json


class TransitionWriter:
    def __init__(self,folder,split,count,origin,metadata,worker=0):
        self.folder=folder;folder.mkdir(parents=True,exist_ok=True)
        self.split=split;self.worker=int(worker);self.count=count;self.origin=origin.cpu().numpy()
        self.buffer={};self.shards=[];self.total=0;self.metadata=metadata
        self.pool=ThreadPoolExecutor(max_workers=1);self.pending=None

    def before(self,state,replay,command,reference,action,requested,policy_action,episode,switched,raw_target_next=None):
        array=lambda x:x.detach().cpu().numpy().copy()
        row={k:array(state[v]) for k,v in {'s':'s','q':'q','dq':'dq','kp':'kp','root_pos':'root_p',
              'root_quat':'root_q','privileged_root':'privileged','ang_vel':'w'}.items()}
        row.update(action=array(action),requested_action=array(requested),policy_action=array(policy_action),
            prev_action=array(replay.hist['action'][:,-1]),cmd_kp=array(command),ref_kp=array(reference),
            episode=episode.copy(),motion_id=replay.command.motion_ids.numpy().copy(),
            t=replay.command.time_steps.numpy().copy(),switch_event=switched.copy())
        if raw_target_next is not None:row['next_raw_kp']=array(raw_target_next)
        self.row=row

    def after(self,state,reference,done,fall):
        array=lambda x:x.detach().cpu().numpy().copy()
        row=self.row
        row.update(next_s=array(state['s']),ach_kp=array(state['kp']),next_root_pos=array(state['root_p']),
            next_root_quat=array(state['root_q']),next_privileged_root=array(state['privileged']),
            next_ref_kp=array(reference),done=array(done),fall=array(fall))
        for key in ('root_pos','next_root_pos'):row[key]-=self.origin
        for key in ('kp','ach_kp','cmd_kp','ref_kp','next_ref_kp','next_raw_kp'):
            if key not in row:continue
            row[key][...,:3]-=self.origin.reshape((self.count,)+(1,)*(row[key].ndim-2)+(3,))
        for k,v in row.items():self.buffer.setdefault(k,[]).append(v)
        self.total+=self.count
        if len(self.buffer['s'])>=64:self.flush()

    def flush(self):
        if not self.buffer:return
        if self.pending is not None:self.pending.result()
        dest=self.folder/f'{self.split}_worker{self.worker:02d}_shard{len(self.shards):04d}.npz'
        if dest.exists():raise FileExistsError(dest)
        arrays={k:np.stack(v) for k,v in self.buffer.items()};self.buffer={}
        self.shards.append(str(dest));self.pending=self.pool.submit(np.savez_compressed,dest,**arrays)

    def close(self):
        self.flush()
        if self.pending is not None:self.pending.result()
        self.pool.shutdown()
        save_json(self.folder/f'{self.split}_worker{self.worker:02d}_manifest.json',dict(self.metadata,
            shards=self.shards,transitions=self.total,num_envs=self.count,worker=self.worker,
            no_cross_reset_sequences=True,body_s_next_s_ground_truth=True,
            next_raw_kp='if present: uncompensated playing K* token1 at t+20ms, for paired G tracking audit; next_ref_kp remains original motion ground truth',
            policy_history_action='previous requested action, including sigma0.05; physical action stored separately',
            reference='actual original full acceptance, retained packets polled50Hz, PACE/RTC; no TRAIN hold override in Isaac'))
