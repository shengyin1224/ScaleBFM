"""T2 causal-input direction screen. No robot and no future commands as NN inputs."""
import json,time
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,digest
from exec_comp.round5.data import discover,load_run
from exec_comp.round5.history import resample
from exec_comp.round5.engine import CurveExecutor,Curve,rotate,qinv,qblend

FOLDER=ROOT/'runs/round5/direction'


def metrics(e,delta):
    if not len(e):return {'count':0,'status':'N/A: no certified samples'}
    en=np.linalg.norm(e,axis=-1);dn=np.linalg.norm(delta,axis=-1);valid=(en>1e-6)&(dn>1e-8)
    cosine=np.divide((e*delta).sum(-1),en*dn,out=np.full(len(e),np.nan),where=valid)
    ratio=np.linalg.norm(e-delta,axis=-1)/np.maximum(en,1e-9)
    amp=dn/np.maximum(en,1e-9)
    def dist(x):
        x=x[np.isfinite(x)]
        if not len(x):return None
        return dict(mean=float(np.mean(x)),p05=float(np.percentile(x,5)),p25=float(np.percentile(x,25)),median=float(np.median(x)),p75=float(np.percentile(x,75)),p95=float(np.percentile(x,95)))
    return dict(count=len(e),cosine=dist(cosine),zero_delta_fraction=float(np.mean(dn<=1e-8)),
        amplitude_ratio=dist(amp),first_order_norm_ratio=dist(ratio),
        mean_error_cm=float(en.mean()*100),mean_first_order_error_cm=float(np.linalg.norm(e-delta,axis=-1).mean()*100),
        pooled_first_order_ratio=float(np.linalg.norm(e-delta,axis=-1).sum()/en.sum()),
        caveat='e-delta assumes unit translational response; not measured corrected robot performance')


def interpolate(times,values,query):
    flat=values.reshape(len(values),-1)
    return np.stack([np.interp(query,times,flat[:,i]) for i in range(flat.shape[1])],-1).reshape((len(query),)+values.shape[1:])


def compute_run(item,models):
    r=load_run(item);plan=r['plan'];t=r['trace_times'];N=len(t)
    if not N:raise ValueError('No active trace')
    future=t+.02
    valid=r['valid'].copy() & (future<=t[-1]) & (np.r_[np.diff(t)>0,False])
    if plan.arrival is not None:valid &= future<plan.arrival
    actual=interpolate(t,r['trace_actual'],future);target=plan.query(future)[:,2:4,:3]
    rootp=interpolate(r['times'],r['p'],future)
    qi=np.searchsorted(r['times'],future).clip(1,len(r['times'])-1);lo=qi-1
    alpha=((future-r['times'][lo])/np.maximum(r['times'][qi]-r['times'][lo],1e-9)).clip(0,1)[:,None]
    rq=qblend(r['quat'][lo],r['quat'][qi],alpha);inverse=qinv(rq)[:,None]
    error=rotate(inverse,target-actual);targetroot=rotate(inverse,target-rootp[:,None]);actualroot=rotate(inverse,actual-rootp[:,None])
    initial=max(0,int(np.searchsorted(plan.stamps,t[0],side='right')-1))
    events=[(t[0],initial)]+[(e['stamp'],j) for j,e in enumerate(plan.chunks) if e['stamp']>t[0] and e['stamp']<=t[-1]]
    states=[(int(ts*1e9),dict(q=r['q'][k],dq=r['dq'][k],kp=r['kp'][k],root_pos=r['p'][k],root_quat=r['quat'][k],w=r['w'][k],g=r['g'][k])) for k,ts in enumerate(r['times'])]
    outputs={};checks=[]
    for method,model in models.items():
        engine=CurveExecutor();delta=np.zeros((N,2,3));proposed=np.zeros_like(delta);eventid=0;pending=None;network_times=[]
        for i,stamp in enumerate(t):
            while eventid<len(events) and events[eventid][0]<=stamp:
                eventtime,index=events[eventid];eventid+=1
                if plan.arrival is not None and eventtime>=plan.arrival:continue
                end=np.searchsorted(r['times'],eventtime,side='right')
                rows=states[max(0,end-40):end]
                if not rows:continue
                history,info=resample(rows,int(eventtime*1e9));goal=plan.window(index,eventtime)
                h={k:torch.from_numpy(v) for k,v in history.items()};w=torch.from_numpy(goal)
                start=time.perf_counter_ns()
                with torch.inference_mode():coef=model(h,w).numpy().copy()
                network_times.append((time.perf_counter_ns()-start)/1e6)
                pending=Curve(coef,int(eventtime*1e9),plan.chunks[index]['sequence'])
            # Nonblocking delivery assumption: installed on the next recorded
            # control tick, never before its source event+1ms CPU compute budget.
            if pending is not None and stamp*1e9>=pending.stamp_ns+1_000_000:
                engine.install(pending,int(stamp*1e9));pending=None
            current=plan.query([stamp])[0]
            if not np.isfinite(current).all():continue
            grasp=bool(r['focus'][i])
            safe,telemetry=engine.step(int(stamp*1e9),np.repeat(current[None],6,0),np.repeat(current[None],6,0),grasp=grasp)
            delta[i]=safe[0,2:4,:3]-current[2:4,:3]
            if engine.curve is not None:proposed[i]=engine.curve.sample(int(stamp*1e9),[0])[0,2:4,:3]
        outputs[method]=rotate(inverse,delta);outputs[method+'_proposed']=rotate(inverse,proposed)
        checks.append(dict(method=method,network_calls=len(network_times),mean_ms=float(np.mean(network_times)) if network_times else None))
    path=FOLDER/'cache'/(item['name']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,times=future,error=error,target_root=targetroot,actual_root=actualroot,focus=r['focus'],valid=valid,**outputs)
    return dict(name=item['name'],handoff=item['handoff'],total_rows=N,certified_rows=int(valid.sum()),
        focus_rows=int(r['focus'].sum()),certified_focus_rows=int((valid&r['focus']).sum()),
        first_handoff_arrival_wall=plan.arrival,
        fk_alignment_mm=dict(median=float(np.median(r['sensorfit'])*1000),p95=float(np.percentile(r['sensorfit'],95)*1000)),
        target_alignment_mm=dict(median=float(np.nanmedian(r['targetfit'])*1000),p95=float(np.nanpercentile(r['targetfit'],95)*1000)),calls=checks)


def decompose(arrays):
    lags=np.arange(-.2,.501,.02);correlations=[]
    for hand in range(2):
        dot=np.zeros(len(lags));aa=np.zeros(len(lags));bb=np.zeros(len(lags))
        for a in arrays:
            ids=np.flatnonzero(a['valid'])
            if len(ids)<30:continue
            t=a['times'][ids];target=a['target_root'][ids,hand];actual=a['actual_root'][ids,hand]
            for j,lag in enumerate(lags):
                shifted=interpolate(t,target,t-lag)
                vt=np.diff(shifted,axis=0);va=np.diff(actual,axis=0)
                mask=(np.diff(t)<.08)&(t[1:]-lag>=t[0])&(t[1:]-lag<=t[-1])
                x=vt[mask];y=va[mask]
                dot[j]+=(x*y).sum();aa[j]+=(x*x).sum();bb[j]+=(y*y).sum()
        score=dot/np.maximum(np.sqrt(aa*bb),1e-12);best=int(np.argmax(score));lag=float(lags[best])
        errors=[];components=[]
        for a in arrays:
            mask=a['valid'];t=a['times'];target=a['target_root'][:,hand]
            shifted=interpolate(t,target,t-lag);errors.append(a['error'][mask,hand]);components.append((target-shifted)[mask])
        e=np.concatenate(errors);lagpart=np.concatenate(components);bias=e.mean(0);lagpart-=lagpart.mean(0)
        total=np.mean(np.sum(e*e,axis=-1));center=e-bias;remaining=center-lagpart
        bias_energy=float(np.dot(bias,bias));lag_gain=float(np.mean(np.sum(center*center,axis=-1))-np.mean(np.sum(remaining*remaining,axis=-1)))
        correlations.append(dict(hand=('left','right')[hand],fixed_offset_cm=(bias*100).tolist(),lag_s=lag,
            velocity_cross_correlation=float(score[best]),mse_fraction_fixed=bias_energy/total,
            mse_fraction_lag=lag_gain/total,mse_fraction_remaining=float(np.mean(np.sum(remaining*remaining,axis=-1))/total),
            remaining_rms_cm=float(np.sqrt(np.mean(np.sum(remaining*remaining,axis=-1)))*100),
            note='Sequential descriptive decomposition; remaining includes unmodelled dynamics, not proven random noise; lag gain may be negative'))
    return correlations


def main():
    FOLDER.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    models={m:torch.jit.load(str(ROOT/'releases/round5'/f'{m}_coefficients_cpu.pt'),map_location='cpu').eval() for m in ('a1','b1')}
    from exec_comp.round4.export import fixture
    h,w=fixture()
    with torch.inference_mode():
        for model in models.values():
            for _ in range(50):model(h,w)
    runs=discover();audit=[]
    for n,item in enumerate(runs):
        path=FOLDER/'cache'/(item['name']+'.npz');receipt=FOLDER/(item['name']+'.json')
        if not receipt.exists():
            result=compute_run(item,models);save_json(receipt,result)
        else:result=json.loads(receipt.read_text())
        audit.append(result);print(n+1,len(runs),item['name'],'certified',result['certified_rows'],'/',result['total_rows'],flush=True)
    arrays=[dict(np.load(FOLDER/'cache'/(r['name']+'.npz'))) for r in runs]
    # Whole-run, stratified train/test split: no frames from the same run on both sides.
    train=set()
    for enabled in (True,False):
        group=[i for i,r in enumerate(runs) if r['handoff']==enabled];train.update(group[::2])
    test=[i for i in range(len(runs)) if i not in train]
    fitted=np.concatenate([a['error'][a['valid']] for i,a in enumerate(arrays) if i in train]).mean(0)
    table={}
    for scope,ids in [('certified_all_runs',list(range(len(runs)))),('heldout_test_runs',test)]:
        table[scope]={}
        for method in ('a1','b1','fixed_offset'):
            table[scope][method]={}
            for phase in ('all','focus'):
                table[scope][method][phase]={}
                for hand in range(2):
                    es=[];ds=[]
                    for i in ids:
                        a=arrays[i];mask=a['valid']&(a['focus'] if phase=='focus' else True)
                        es.append(a['error'][mask,hand]);ds.append(np.broadcast_to(fitted[hand],(int(mask.sum()),3)) if method=='fixed_offset' else a[method][mask,hand])
                    table[scope][method][phase][('left','right')[hand]]=metrics(np.concatenate(es),np.concatenate(ds))
    full_complete=all(a['certified_rows']==a['total_rows'] for a in audit)
    result=dict(date='2026-10-26',requested_runs=46,run_audit=audit,table=table,
        full_corpus_direction='N/A: uncertified missing overlay/sensor-alignment samples remain' if not full_complete else 'complete',
        covered_rows=sum(a['certified_rows'] for a in audit),total_rows=sum(a['total_rows'] for a in audit),
        training_runs=[runs[i]['name'] for i in sorted(train)],test_runs=[runs[i]['name'] for i in test],fixed_offset_fit_cm=(fitted*100).tolist(),
        decomposition=decompose(arrays),decision='DO NOT RUN ROBOT: full-corpus/FOCUS reconstruction is incomplete; inspect heldout direction before any promotion',
        target_definition='K*(t+20ms)-actualFK(t+20ms), world difference rotated by measured root quaternion(t+20ms); delta generated causally at t, transformed into same future root frame',
        reconstruction='Uses accepted aligned chunks and acceptance phase/blend, no future replans in event inputs. Only timestamps before first handoff arrival and<=5mm current target/FK mismatch are certified.',
        limitations=['Unserialized per-hand playback/hold/rejoin and early old sparse phase logs prevent whole-corpus certification.',
            'Fixed20ms response assumption is a diagnostic, not an identified robot impulse response.',
            'Nonblocking worker delivery is represented as next control tick after1ms compute; hardware scheduling evaluated separately in T1.'],real_robot_run=False)
    save_json(FOLDER/'summary.json',result)


if __name__=='__main__':main()
