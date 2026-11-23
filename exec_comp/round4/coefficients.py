"""B1 native offset/slope in world metres, before spherical clipping."""
import torch
from exec_comp.models.amortized import encode_state
from exec_comp.geometry import heading_anchor,to_anchor,qapply
from exec_comp.common import CHUNK,CHUNK_FPS

SPAN=(CHUNK-1)/CHUNK_FPS


def native_coefficients(predictor,history,window):
    if hasattr(predictor,'predict_coefficients'):
        return predictor.predict_coefficients(history,window)
    model=predictor.model
    if getattr(model,'parameterization',None)!='lowdim':raise ValueError('Coefficient transition requires native lowdim B1')
    origin,heading=heading_anchor(history['root_pos'],history['root_quat'])
    state=history['s']
    encoded=encode_state(to_anchor(history['kp'],origin,heading),history['q'],history['dq'],state[...,58:61],state[...,61:64])
    raw=model.parameters_for(encoded,to_anchor(window,origin,heading)).view(-1,2,4,3)
    coefficients=torch.tanh(raw)*model.residual_scale_m
    return qapply(heading[:,None,None].expand(*coefficients.shape[:-1],4),coefficients)


def advance(coefficients,seconds):
    result=coefficients.clone()
    result[:,0]+=result[:,1]*(seconds[:,None,None]/SPAN)
    return result


def expand(coefficients,frames=CHUNK):
    phase=torch.linspace(0.,1.,frames,device=coefficients.device,dtype=coefficients.dtype)[None,:,None,None]
    value=coefficients[:,0,None]+coefficients[:,1,None]*phase
    return value*torch.minimum(torch.ones_like(value[...,:1]),.05/value.norm(dim=-1,keepdim=True).clamp_min(1e-8))
