"""Episode-safe sequence sampling from shared simulation rollouts."""
import json
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import ROOT,HISTORY,CHUNK,CHUNK_FPS
from exec_comp.geometry import to_anchor,pose_normalize,heading_anchor


class Rollouts:
    def __init__(self,split,body=False,device="cuda:0",folders=None,balance_roots=False,keys=None):
        self.device=device;self.body=body;self.streams=[]
        roots=[Path(p) for p in folders] if folders else [ROOT/'runs/data']
        manifests=[p for root in roots for p in sorted(root.glob(split+"_worker*_manifest.json"))]
        if not manifests:raise FileNotFoundError(f"No completed {split} data manifests")
        # `keys` lets a caller read a different field set from the same shards;
        # None keeps the two original selections byte for byte.
        keys=tuple(keys) if keys is not None else (
              ("s","next_s","action","episode","done","root_pos","root_quat","next_root_pos","next_root_quat",
              "privileged_root","next_privileged_root","kp","ach_kp") if body else (
              "q","dq","kp","ach_kp","cmd_kp","ref_kp","root_pos","root_quat","episode","done"))
        self.total=0
        for path in manifests:
            manifest=json.loads(path.read_text());pieces={k:[] for k in keys}
            for shard in manifest["shards"]:
                with np.load(shard,allow_pickle=False) as f:
                    for k in keys:
                        value=f[k]
                        if k=="cmd_kp":value=value[:,:,0]  # Actual command at that control step.
                        pieces[k].append(value)
            arrays={k:np.swapaxes(np.concatenate(v,axis=0),0,1).copy() for k,v in pieces.items()}
            self.streams.append(arrays);self.total+=arrays["episode"].size
        self.rng=np.random.default_rng(42)
        self.weights=np.asarray([s["episode"].size for s in self.streams],dtype=float);self.weights/=self.weights.sum()
        if balance_roots:
            for root in roots:
                indices=[i for i,p in enumerate(manifests) if p.parent==root]
                if not indices:raise FileNotFoundError(f'No {split} manifests for balanced root {root}')
                total=sum(self.streams[i]['episode'].size for i in indices)
                for i in indices:self.weights[i]=self.streams[i]['episode'].size/total/len(roots)

    def indices(self,batch,horizon,history=HISTORY):
        stream=self.streams[int(self.rng.choice(len(self.streams),p=self.weights))]
        envs,steps=stream["episode"].shape
        if steps<=history+horizon+1:raise ValueError("Rollout is shorter than required horizon")
        good_e=[];good_t=[];count=0
        for _ in range(100):
            e=self.rng.integers(envs,size=batch*2)
            t=self.rng.integers(history-1,steps-horizon-1,size=batch*2)
            valid=(stream["episode"][e,t-history+1]==stream["episode"][e,t+horizon])&(~stream["done"][e,t+horizon])
            good_e.append(e[valid]);good_t.append(t[valid]);count+=int(valid.sum())
            if count>=batch:break
        if count<batch:raise ValueError("Not enough within-episode windows")
        return stream,np.concatenate(good_e)[:batch],np.concatenate(good_t)[:batch]

    def tensor(self,array):
        return torch.as_tensor(array,device=self.device,dtype=torch.float32)

    def keypoint_batch(self,batch,goal="ach",chunk=CHUNK):
        offsets=np.arange(chunk)*50/CHUNK_FPS
        lower=offsets.astype(int);upper=np.ceil(offsets).astype(int)
        stream,e,t=self.indices(batch,int(upper[-1]))
        hist=t[:,None]+np.arange(-HISTORY+1,1)[None]
        root=self.tensor(stream["root_pos"][e,t]);quat=self.tensor(stream["root_quat"][e,t])
        root,quat=heading_anchor(root,quat)
        kp=to_anchor(self.tensor(stream["kp"][e[:,None],hist]),root,quat)
        q=self.tensor(stream["q"][e[:,None],hist]);dq=self.tensor(stream["dq"][e[:,None],hist])
        state=torch.cat((kp.flatten(2),q,dq),-1).flatten(1)
        w=self.tensor(offsets-lower)[None,:,None,None]
        def future(key):
            lo=self.tensor(stream[key][e[:,None],t[:,None]+lower])
            hi=self.tensor(stream[key][e[:,None],t[:,None]+upper])
            return to_anchor(pose_normalize(lo+(hi-lo)*w),root,quat)
        desired=future("ach_kp" if goal=="ach" else "ref_kp")
        cmd=future("cmd_kp")
        ach=future("ach_kp")
        return {"state":state,"goal":desired,"cmd":cmd,"ach":ach}

    def body_batch(self,batch,horizon=8,privileged=False):
        stream,e,t=self.indices(batch,horizon,history=1)
        offsets=np.arange(horizon)
        s=self.tensor(stream["s"][e[:,None],t[:,None]+offsets])
        ns=self.tensor(stream["next_s"][e[:,None],t[:,None]+offsets])
        if privileged:
            s=torch.cat((s,self.tensor(stream["privileged_root"][e[:,None],t[:,None]+offsets])),-1)
            ns=torch.cat((ns,self.tensor(stream["next_privileged_root"][e[:,None],t[:,None]+offsets])),-1)
        return {"s":s,"next_s":ns,"action":self.tensor(stream["action"][e[:,None],t[:,None]+offsets]),
                "root_pos":self.tensor(stream["root_pos"][e,t]),"root_quat":self.tensor(stream["root_quat"][e,t]),
                "next_root_pos":self.tensor(stream["next_root_pos"][e[:,None],t[:,None]+offsets]),
                "next_root_quat":self.tensor(stream["next_root_quat"][e[:,None],t[:,None]+offsets]),
                "ach":self.tensor(stream["ach_kp"][e[:,None],t[:,None]+offsets])}
