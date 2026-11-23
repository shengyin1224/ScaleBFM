"""HAT uses two matrix ROWS; the cerebellum uses x/z COLUMNS. Never copy 6D directly."""
import torch
from exec_comp.geometry import matrix_to_q,qnormalize,qmul,qconj,qapply,pose9


def hat_rotation_quat(value):
    if torch.any(value[...,:3].norm(dim=-1)<1e-6):raise ValueError('Degenerate first HAT rotation row')
    x=torch.nn.functional.normalize(value[...,:3],dim=-1)
    y=value[...,3:]-(value[...,3:]*x).sum(-1,keepdim=True)*x
    if torch.any(y.norm(dim=-1)<1e-6):raise ValueError('Collinear HAT rotation rows')
    y=torch.nn.functional.normalize(y,dim=-1)
    return matrix_to_q(torch.stack((x,y,torch.linalg.cross(x,y,dim=-1)),-2))


def action_to_keypoints(action):
    action=torch.as_tensor(action,dtype=torch.float32)
    poses=[]
    for index,start in enumerate((89,0,80,30)):
        block=action[...,start:start+9];p=block[...,:3];q=hat_rotation_quat(block[...,3:])
        if index>=2:
            mount=q.new_tensor([.70710678,0,0,.70710678] if index==2 else [0,.70710678,-.70710678,0])
            q=qnormalize(qmul(q,qconj(mount.expand_as(q))))
            p=p-qapply(q,q.new_tensor([.0415,0,0]).expand_as(p))
        poses.append(pose9(p,q))
    return torch.stack(poses,-2)
