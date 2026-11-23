"""Measured state cache: timestamp interpolation happens only in the worker."""
from collections import deque
import numpy as np
from exec_comp.round5.engine import qblend,rotate,qinv


class TimestampHistory:
    def __init__(self,duration_s=.5):self.duration_ns=int(duration_s*1e9);self.rows=deque()
    def append(self,stamp_ns,state):
        if self.rows and stamp_ns<=self.rows[-1][0]:return
        self.rows.append((int(stamp_ns),state))
        while len(self.rows)>2 and stamp_ns-self.rows[1][0]>self.duration_ns:self.rows.popleft()
    def snapshot(self):return list(self.rows)


def resample(rows,stamp_ns):
    if not rows:raise ValueError('No measured states')
    times=np.asarray([r[0] for r in rows],dtype=np.int64)
    query=int(stamp_ns)+np.arange(-9,1,dtype=np.int64)*20_000_000
    hi=np.searchsorted(times,query).clip(0,len(times)-1);lo=(hi-1).clip(0,len(times)-1)
    fraction=np.divide(query-times[lo],np.maximum(times[hi]-times[lo],1)).clip(0,1)
    output={}
    for key in ('q','dq','kp','root_pos','root_quat','w','g'):
        values=np.stack([r[1][key] for r in rows]);a,b=values[lo],values[hi]
        alpha=fraction.reshape((-1,)+(1,)*(a.ndim-1))
        if key=='root_quat':output[key]=qblend(a,b,alpha)
        else:output[key]=a+(b-a)*alpha
    # kp contains6D rotation axes; orthonormalize the interpolated columns.
    x=output['kp'][...,3:6];z=output['kp'][...,6:9]
    x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    z=z-(z*x).sum(-1,keepdims=True)*x;z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
    output['kp'][...,3:6]=x;output['kp'][...,6:9]=z
    output['s']=np.concatenate((output['q'],output['dq'],output['w'],output['g']),-1)
    return {k:np.ascontiguousarray(v,dtype=np.float32) for k,v in output.items()},dict(
        measured_history_span_s=float((times[-1]-times[0])*1e-9),requested_history_span_s=.18,
        left_padded_samples=int(np.count_nonzero(query<times[0])),right_clamped_samples=int(np.count_nonzero(query>times[-1])))
