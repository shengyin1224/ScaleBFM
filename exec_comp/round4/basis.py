"""Exact least-squares temporal bases on the existing100-frame A1 grid."""
import numpy as np
import torch
from scipy.interpolate import BSpline

NAMES=('ramp','dct3','dct4','dct6','bspline4','bspline6')


def design(name,frames=100):
    t=np.linspace(0.,1.,frames)
    if name=='ramp':matrix=np.stack((np.ones_like(t),t),1)
    elif name.startswith('dct'):
        k=int(name[3:]);matrix=np.cos(np.pi*(np.arange(frames)[:,None]+.5)*np.arange(k)[None]/frames)
    elif name.startswith('bspline'):
        m=int(name[7:]);degree=3
        knots=np.r_[np.zeros(degree+1),np.linspace(0,1,m-degree+1)[1:-1],np.ones(degree+1)]
        matrix=BSpline.design_matrix(t,knots,degree,extrapolate=False).toarray()
    else:raise ValueError(name)
    if np.linalg.matrix_rank(matrix)!=matrix.shape[1]:raise ValueError('Rank deficient basis')
    return matrix


class TemporalBasis(torch.nn.Module):
    def __init__(self,name,frames=100):
        super().__init__();self.name=name
        matrix=design(name,frames)
        self.register_buffer('matrix',torch.tensor(matrix,dtype=torch.float32))
        self.register_buffer('inverse',torch.tensor(np.linalg.pinv(matrix),dtype=torch.float32))

    def encode(self,values):return torch.einsum('kt,btnc->bknc',self.inverse,values)
    def decode(self,coefficients):return torch.einsum('tk,bknc->btnc',self.matrix,coefficients)
    def forward(self,values):return self.decode(self.encode(values))


def rotation_log(quaternion):
    q=torch.nn.functional.normalize(quaternion,dim=-1)
    q=torch.where(q[...,:1]<0,-q,q)
    v=q[...,1:];length=v.norm(dim=-1,keepdim=True)
    angle=2*torch.atan2(length,q[...,:1])
    return v*(angle/length.clamp_min(1e-9))


def rotation_exp(vector):
    angle=vector.norm(dim=-1,keepdim=True)
    return torch.cat((torch.cos(angle*.5),vector*.5*torch.sinc(angle/(2*torch.pi))),-1)
