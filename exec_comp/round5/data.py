"""Audit/reconstruct26-Oct raw records; never substitutes future commands for NN input."""
from pathlib import Path
from datetime import datetime
import re,json
import yaml
import numpy as np
from exec_comp.common import ROOT,PROJECT,save_json
from exec_comp.round5.reference import canonical7

DAY='2026-10-26'
OUTPUTS=PROJECT/'ScaleBridge/outputs'/DAY
RECORDS=Path('/home/nerv/qingyaoxu/human_policy/online_hat_records')


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            try:yield json.loads(line)
            except json.JSONDecodeError:continue  # final partially-written row only; audit count separately


def discover():
    output=[]
    for d in sorted(OUTPUTS.iterdir()):
        cfg=d/'.hydra/config.yaml';trace=d/'wrist_compensation_trace.jsonl'
        if not cfg.exists() or not trace.exists() or trace.stat().st_size==0:continue
        config=yaml.safe_load(cfg.read_text());e=config['env']['config']
        choices=yaml.safe_load((d/'.hydra/hydra.yaml').read_text())['hydra']['runtime']['choices']
        if choices.get('env')!='motion_tracking_hat4_adaptive' or config.get('exec_comp',{}).get('enable',False):continue
        if e.get('wrist_compensation','none')!='none' or e.get('reference_forcing') or e.get('min_chunk_execution_fraction')!=.1:continue
        output.append(dict(name=d.name,path=str(d),handoff=bool(e.get('focus_handoff_enabled')),checkpoint=config['agent']['config']['checkpoint']))
    return output


class RecordedPlan:
    def __init__(self,folder):
        from scalebridge.online.reference import sample_chunk
        self.sampler=sample_chunk;self.folder=Path(folder)
        log=(self.folder/'run.log').read_text()
        frames={int(m[1]):float(m[2]) for line in log.splitlines()
            if (m:=re.search(r'Accepted chunk (\d+) .*?:.*? frame ([0-9.]+), source_age=',line))}
        self.chunks=[]
        for row in jsonlines(self.folder/'online_hat_chunks.jsonl'):
            c={k:np.asarray(v) if isinstance(v,list) else v for k,v in row['aligned'].items()}
            seq=int(c['sequence_id'])
            if seq in frames:self.chunks.append(dict(chunk=c,stamp=float(row['time']),start=frames[seq],sequence=seq))
        self.stamps=np.asarray([r['stamp'] for r in self.chunks])
        self.arrival=None
        for line in log.splitlines():
            if re.search(r'Actual (left|right) wrist reached .* entry;',line):
                self.arrival=datetime.fromisoformat(line[:23]).timestamp();break
        self.starts=[datetime.fromisoformat(line[:23]).timestamp() for line in log.splitlines() if 'R1: starting closed loop.' in line]
        self.config=yaml.safe_load((self.folder/'.hydra/config.yaml').read_text())['env']['config']

    def query(self,times,indices=None):
        from scalebridge.online.reference import quat_slerp
        times=np.atleast_1d(times).astype(float)
        ids=np.searchsorted(self.stamps,times,side='right')-1 if indices is None else np.broadcast_to(indices,times.shape)
        poses=np.full((len(times),4,7),np.nan,dtype=np.float32)
        for idx in np.unique(ids):
            if idx<0:continue
            mask=ids==idx;event=self.chunks[int(idx)];c=event['chunk']
            offsets=event['start']/c['fps']+times[mask]-event['stamp']
            sample=self.sampler(c,0.,offsets)
            if idx:
                old=self.chunks[int(idx)-1];pc=old['chunk'];prior=self.sampler(pc,0.,old['start']/pc['fps']+times[mask]-old['stamp'])
                alpha=np.clip((times[mask]-event['stamp'])/float(self.config.get('chunk_blend_s',.22)),0,1)
                for key in sample:
                    if key=='focus_phase':continue
                    a=alpha.reshape((-1,)+(1,)*(sample[key].ndim-1))
                    sample[key]=quat_slerp(prior[key],sample[key],alpha).astype(np.float32) if 'quat' in key else (prior[key]*(1-a)+sample[key]*a).astype(np.float32)
            poses[mask]=canonical7(sample)
        return poses

    def window(self,index,now=None):
        # Reproduce ORIGINAL event-time sampler: the cross-chunk blend fraction
        # is fixed at this event time for all future tokens, as in R4 hooks.
        from scalebridge.online.reference import quat_slerp
        event=self.chunks[index];c=event['chunk'];now=event['stamp'] if now is None else now
        sample=self.sampler(c,event['start']/c['fps']+now-event['stamp'],np.arange(100)/30)
        alpha=np.clip((now-event['stamp'])/max(float(self.config.get('chunk_blend_s',.22)),1e-9),0,1)
        if index and alpha<1:
            old=self.chunks[index-1];pc=old['chunk']
            prior=self.sampler(pc,old['start']/pc['fps']+now-old['stamp'],np.arange(100)/30)
            for key in sample:
                if key=='focus_phase':continue
                sample[key]=quat_slerp(prior[key],sample[key],np.full(100,alpha)).astype(np.float32) if 'quat' in key else (prior[key]*(1-alpha)+sample[key]*alpha).astype(np.float32)
        p=canonical7(sample)
        from exec_comp.round3.metrics import quat_columns
        x,z=quat_columns(p[...,3:]);return np.ascontiguousarray(np.concatenate((p[...,:3],x,z),-1),dtype=np.float32)


def load_run(item):
    import torch
    from scipy.spatial import cKDTree
    from exec_comp.diff.fk import FK
    from exec_comp.geometry import from_anchor
    folder=Path(item['path']);plan=RecordedPlan(folder)
    controls=[r for r in jsonlines(folder/'real_world_control.jsonl') if r.get('record_type')=='control_frame' and r.get('global_root_pos') is not None]
    trace=list(jsonlines(folder/'wrist_compensation_trace.jsonl'))
    ts=np.array([r['time'] for r in controls]);q=np.array([r['dof_pos'] for r in controls],dtype=np.float32)
    dq=np.array([r['dof_vel'] for r in controls],dtype=np.float32);p=np.array([r['global_root_pos'] for r in controls],dtype=np.float32)
    quat=np.array([r['root_quat_wxyz'] for r in controls],dtype=np.float32);w=np.array([r['base_ang_vel'] for r in controls],dtype=np.float32)
    with torch.no_grad():kp=from_anchor(FK().keypoints(torch.from_numpy(q)),torch.from_numpy(p),torch.from_numpy(quat)).numpy()
    measured=np.array([[r['meas_lw'],r['meas_rw']] for r in trace],dtype=np.float32)
    rootmeas=np.array([r['meas_root'] for r in trace],dtype=np.float32)
    features=np.concatenate((p,kp[:,2:4,:3].reshape(-1,6)),axis=1)
    wanted=np.concatenate((rootmeas,measured.reshape(-1,6)),axis=1)
    first=int(np.searchsorted(ts,plan.starts[0],side='left')) if plan.starts else max(0,len(ts)-len(trace))
    # One trace row is emitted after each active control action. Do NOT let a
    # spatial nearest-neighbour search jump forward through a stationary hold:
    # that silently changes timestamps and destroys lag estimates.
    chosen=np.clip(first+np.arange(len(trace)),0,len(ts)-1)
    chosen=np.asarray(chosen);t=ts[chosen]
    rootq=quat[chosen];ref=np.array([[r['ref_lw'],r['ref_rw']] for r in trace]);focus=np.array([r['focus']>=.5 for r in trace])
    predicted=plan.query(t)
    fit=np.linalg.norm(predicted[:,2:4,:3]-ref,axis=-1)
    sensorfit=np.linalg.norm(kp[chosen,2:4,:3]-measured,axis=-1)
    # Clock stamps in chunk recording occur after acceptance, and sensor SDK
    # reads are asynchronous. Keep all samples in diagnostics; certify only
    # windows with<=5mm current-position mismatch and pre-handoff state.
    before=np.ones(len(t),dtype=bool) if plan.arrival is None else t<plan.arrival
    valid=before & (sensorfit.max(-1)<=.005) & (fit.max(-1)<=.005) & np.isfinite(predicted).all((1,2))
    from exec_comp.round5.engine import rotate,qinv
    g=rotate(qinv(quat),np.broadcast_to(np.array([0.,0.,-1.]),(len(quat),3)))
    return dict(item=item,plan=plan,controls=controls,times=ts,q=q,dq=dq,p=p,quat=quat,w=w,g=g,kp=kp,
        trace_times=t,trace_actual=measured,trace_ref=ref,focus=focus,rootq=rootq,valid=valid,
        sensorfit=sensorfit,targetfit=fit)


def audit():
    runs=discover();result=dict(date=DAY,runs=runs,handoff_on=sum(r['handoff'] for r in runs),handoff_off=sum(not r['handoff'] for r in runs),
        target_reconstruction='Accepted aligned chunks including RTC; phase from acceptance logs; original blend semantics. No future replan is used as network input.',
        limitations=['Per-hand handoff playback/rejoin pose state was not serialized. Samples after first arrival are not certified and whole-corpus/FOCUS conclusions must report coverage/N/A.',
            'Sensor q/dq snapshots are asynchronous to wrist trace; reconstructed current FK and target mismatches are quantified and must not be hidden.'])
    save_json(ROOT/'runs/round5/data_audit.json',result);print('raw',len(runs),result['handoff_on'],result['handoff_off'])
    return result


if __name__=='__main__':audit()
