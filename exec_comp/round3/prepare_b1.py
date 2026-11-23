"""Cache actual deployment-sampled training targets, with explicit event oversampling.

Two natural PACE switches cannot fit into17steps. TRAIN augmentation overrides
only the hold eligibility at scheduled events. Matching/RTC/overlap/sampling
are the pinned originals. Isaac evaluation and online collection never use the
override. No arbitrary translation noise or duplicated time advancement.
"""
import argparse
import numpy as np
import torch
from exec_comp.common import ROOT,save_json,log,CHUNK
from exec_comp.geometry import pose_normalize
from exec_comp.round3.b1_data import B1Rollouts,deployed_contract,switch_schedule
from exec_comp.round3.switching import DeployedSwitching,make_chunk,quiet_deployment_logging
from exec_comp.round3.metrics import to_links,canonical


def simulate_reference(batch,contract,generator,horizon=32):
    cfg={k:v['value'] for k,v in contract['switching_parameters'].items()}
    fps=float(contract['chunk_fps']);age_steps=round(contract['source_age_s']['p50']/.02)
    all_windows=[];all_tokens=[];all_events=[];forced=0
    for i in range(len(batch['s'])):
        switcher=DeployedSwitching(config=cfg,future_offsets_s=np.arange(6)*.02,anti_stall=True)
        schedule=switch_schedule(generator,17,contract)['steps']
        schedule+= [s+17 for s in switch_schedule(generator,horizon-17,contract)['steps']] if horizon>17 else []
        schedule=[s for s in schedule if s<horizon]
        positions,quats=to_links(batch['source_keypoints'][i]);positions=positions.numpy();quats=quats.numpy()
        source=batch['source_reference'][i];windows=[];tokens=[];events=[]
        for step in range(horizon):
            now=step*20_000_000;sequence=switcher.publish_state(now,positions[step],quats[step])
            accepted=False
            if step==0 or step in schedule:
                start=max(0,step-age_steps) if step else 0
                frames=start+np.arange(CHUNK)*50/fps
                lo=frames.astype(int);hi=np.ceil(frames).astype(int)
                weight=torch.tensor(frames-lo,dtype=source.dtype)[:,None,None]
                candidate=pose_normalize(source[lo]+(source[hi]-source[lo])*weight)
                p,q=to_links(candidate)
                chunk=make_chunk(sequence,max(0,sequence-(step-start)),now,fps,(p.numpy(),q.numpy()))
                forced+=int(step>0 and switcher.hold_active(now))
                accepted=switcher.offer(chunk,now,training_oversample=step>0)['accepted']
            if switcher.env.chunk is None:raise RuntimeError('Training initial chunk rejected')
            tokens.append(canonical(switcher.sample(now)))
            windows.append(canonical(switcher.sample_window(now,np.arange(CHUNK)/30)))
            events.append(accepted)
        if sum(events)-int(events[0])<2:raise RuntimeError('Training scenario lacks two accepted switches')
        all_windows.append(windows);all_tokens.append(tokens);all_events.append(events)
    return {'windows':torch.from_numpy(np.asarray(all_windows)),
            'raw_tokens':torch.from_numpy(np.asarray(all_tokens)),
            'events':torch.tensor(all_events,dtype=torch.bool)},forced


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=int,default=4096)
    p.add_argument('--validation',type=int,default=512);p.add_argument('--name',default='initial')
    p.add_argument('--extra-data',action='append',default=[]);p.add_argument('--balance-data-roots',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);quiet_deployment_logging();generator=np.random.default_rng(20261026)
    folder=ROOT/f'runs/round3/b1_cache/{args.name}';folder.mkdir(parents=True,exist_ok=True)
    contract=deployed_contract();metadata={'horizon':32,'training_hold_oversampled':True,
        'reason':'>=2 switches in17steps conflicts with natural25-50frames@15Hz hold; only TRAIN eligibility is oversampled',
        'matching_RTC_sampling':'hash-pinned original deployment implementation',
        'prediction_noise_m':0,'extra_data':args.extra_data,'splits':{}}
    for split,n in (('train',args.train),('validation',args.validation)):
        destination=folder/f'{split}.pt'
        if destination.exists():raise FileExistsError(destination)
        data=B1Rollouts(split,device='cpu',folders=[ROOT/'runs/data',*args.extra_data],balance_roots=args.balance_data_roots)
        pieces=[];forced=0
        for start in range(0,n,64):
            batch=data.switching_batch(min(64,n-start),32)
            raw,count=simulate_reference(batch,contract,generator,32);forced+=count
            pieces.append({**{k:v for k,v in batch.items() if not k.startswith('source_')},**raw})
            log('b1_cache_progress',split=split,scenarios=min(n,start+64),total=n)
        tensors={k:torch.cat([part[k] for part in pieces]) for k in pieces[0]}
        assert bool((tensors['events'][:,1:17].sum(1)>=2).all())
        torch.save(tensors,destination)
        metadata['splits'][split]={'scenarios':n,'forced_hold_overrides':forced,'file':str(destination)}
        save_json(folder/'manifest.json',metadata);del data,pieces,tensors
    log('b1_cache_complete',folder=str(folder))


if __name__=='__main__':main()
