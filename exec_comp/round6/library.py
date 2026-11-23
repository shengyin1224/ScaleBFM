from pathlib import Path
import json,os
import numpy as np
from exec_comp.common import ROOT,TRACK,metadata
from exec_comp.round5.engine import qblend,rotate

class Clip:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(path,allow_pickle=False) as a:self.data={k:a[k] for k in a.files}
        self.fps=float(self.data['fps']);self.frames=len(self.data['joint_pos']);self.duration=(self.frames-1)/self.fps
        ids=[metadata()['body_names'].index(n) for n in ('pelvis','torso_link','left_wrist_yaw_link','right_wrist_yaw_link')]
        self.pos=self.data['body_pos_w'][:,ids].copy();self.quat=self.data['body_quat_w'][:,ids].copy()
        self.pos[:,1]+=rotate(self.quat[:,1],np.broadcast_to(np.array([0.,0.,.4]),(self.frames,3)))
    def sample(self,times):
        frames=np.asarray(times)*self.fps;frames=np.clip(frames,0,self.frames-1)
        lo=np.floor(frames).astype(int);hi=np.minimum(lo+1,self.frames-1);w=(frames-lo)[:,None,None]
        return self.pos[lo]*(1-w)+self.pos[hi]*w,qblend(self.quat[lo],self.quat[hi],w)
    def reset(self,world,start=10,rng=None):
        a=self.data;i=min(start,self.frames-1);q=a['body_quat_w'][i,0]
        # Recorded ang_vel is world-frame; MuJoCo freejoint angular velocity is local.
        from exec_comp.round5.engine import qinv
        return world.reset(a['joint_pos'][i],a['body_pos_w'][i,0],q,a['joint_vel'][i],a['body_lin_vel_w'][i,0],rotate(qinv(q),a['body_ang_vel_w'][i,0]),rng=rng)
    def focus(self,time):return bool(self.data.get('focus_mask',np.zeros(self.frames,bool))[int(np.clip(time*self.fps,0,self.frames-1))])


def paths(split):
    data=Path(os.environ.get('R6_CORPUS',str(TRACK/'data/hat4_unifolm_focus_v3/processed')))
    ids=json.loads((ROOT/'configs/split.json').read_text())[split]
    return [data/(name+'.npz') for name in ids]


def canonical(p,q):
    x=rotate(q,np.broadcast_to(np.array([1.,0.,0.]),p.shape));z=rotate(q,np.broadcast_to(np.array([0.,0.,1.]),p.shape))
    return np.concatenate((p,x,z),-1).astype(np.float32)
