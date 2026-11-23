"""Bounded NumPy control-side curve expansion. No torch, network, FK or disk IO."""
from dataclasses import dataclass
import numpy as np


def qmul(a,b):
    w=a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True)
    v=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate((w,v),-1)


def qinv(q):return q*np.array([1.,-1.,-1.,-1.])


def qexp(v):
    angle=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.concatenate((np.cos(angle/2),v*.5*np.sinc(angle/(2*np.pi))),-1)


def qlog(q):
    q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    q=np.where(q[...,:1]<0,-q,q);v=q[...,1:];n=np.linalg.norm(v,axis=-1,keepdims=True)
    return v*(2*np.arctan2(n,q[...,:1])/np.maximum(n,1e-12))


def qblend(a,b,alpha):return qmul(qexp(qlog(qmul(b,qinv(a)))*alpha),a)


def rotate(q,v):
    u=np.cross(q[...,1:],v)
    return v+2*(q[...,:1]*u+np.cross(q[...,1:],u))


@dataclass(frozen=True)
class Curve:
    coefficients:np.ndarray  # [2,4,6] in world m/rad, before shared cap
    stamp_ns:int
    sequence:int
    epoch:int=0
    span_s:float=3.3

    def sample(self,now_ns,offsets):
        phase=np.clip(((now_ns-self.stamp_ns)*1e-9+np.asarray(offsets))/self.span_s,0,1)
        return self.coefficients[0][None]+phase[:,None,None]*self.coefficients[1][None]


class CurveExecutor:
    def __init__(self,enable=True,cap_m=.05,max_step_m=.01,angular_step_deg=3.,angular_rate_deg_s=150.,
                 fade_s=.15,transition_s=.12,cap_hold_s=1.,cap_guard='residual',lowpass=True):
        self.enable=enable;self.cap=cap_m;self.max_step=max_step_m;self.angular_step=np.deg2rad(angular_step_deg)
        self.angular_rate=np.deg2rad(angular_rate_deg_s);self.fade_s=fade_s;self.transition_s=transition_s
        self.cap_hold_s=cap_hold_s;self.cap_guard=cap_guard;self.lowpass=lowpass;self.reset()

    def reset(self):
        self.curve=self.previous_curve=None;self.transition_start=None;self.last_ns=None
        self.pos=self.rot=None;self.frozen=False;self.freeze_pos=self.freeze_rot=None
        self.release_start=None;self.fallback_start=None;self.cap_since=None
        self.counters={k:0 for k in ('nan','sustained_cap','step_jump','angular_jump','reference_reset','expired')}
        self.fallback_reason=[];self.last_dt_s=None
        self.initial_curve_pending=False

    def install(self,curve,now_ns):
        if curve.coefficients.shape!=(2,4,6):raise ValueError('Coefficient shape')
        if not np.isfinite(curve.coefficients).all():return False
        self.initial_curve_pending=self.curve is None
        self.previous_curve=self.curve;self.curve=curve;self.transition_start=now_ns
        return True

    def step(self,now_ns,deployed,raw,*,grasp=False,offsets=None,force_fallback=False):
        if not self.enable:return deployed,{}  # exact object identity
        deployed=np.asarray(deployed);raw=np.asarray(raw)
        if offsets is None:offsets=np.arange(len(raw))*.02
        dt=.02 if self.last_ns is None else max(0,(now_ns-self.last_ns)*1e-9)
        if self.last_ns is not None and now_ns<self.last_ns:raise ValueError('Non-monotonic control timestamp')
        self.last_ns=now_ns;self.last_dt_s=dt
        neutral=np.zeros((*raw.shape[:-1],4));neutral[...,0]=1
        if self.pos is None:self.pos=np.zeros_like(raw[...,:3]);self.rot=neutral.copy()
        if self.initial_curve_pending:
            self.freeze_pos=self.pos.copy();self.freeze_rot=self.rot.copy();self.release_start=now_ns
            self.initial_curve_pending=False
        candidate=np.zeros((*raw.shape[:-1],6));expired=False
        if self.curve is not None:
            candidate=self.curve.sample(now_ns,offsets)
            if self.previous_curve is not None and self.transition_start is not None:
                alpha=np.clip((now_ns-self.transition_start)*1e-9/max(self.transition_s,1e-9),0,1)
                candidate=self.previous_curve.sample(now_ns,offsets)*(1-alpha)+candidate*alpha
                if alpha>=1:self.previous_curve=None
            expired=(now_ns-self.curve.stamp_ns)*1e-9>self.curve.span_s
        position=candidate[...,:3];rotation=qexp(candidate[...,3:])
        if self.lowpass:
            # Same alpha=.5 at20ms; physical-time coefficient at other periods.
            state=self.pos[0];values=[]
            for j,value in enumerate(position):
                interval=dt if j==0 else max(float(offsets[j]-offsets[j-1]),0)
                alpha=1-np.exp(-interval/(.02/np.log(2)))
                state=state+(value-state)*alpha;values.append(state)
            position=np.asarray(values)
        if grasp and not self.frozen:
            self.freeze_pos=self.pos.copy();self.freeze_rot=self.rot.copy()
        if not grasp and self.frozen:self.release_start=now_ns
        self.frozen=bool(grasp)
        if grasp:
            position=self.freeze_pos.copy();rotation=self.freeze_rot.copy()
        elif self.release_start is not None:
            alpha=np.clip((now_ns-self.release_start)*1e-9/max(self.fade_s,1e-9),0,1)
            position=self.freeze_pos+(position-self.freeze_pos)*alpha
            rotation=qblend(self.freeze_rot,rotation,alpha)
            if alpha>=1:self.release_start=None
        lookahead=deployed[...,:3]-raw[...,:3];combined=lookahead+position
        norm=np.linalg.norm(combined,axis=-1,keepdims=True);scale=np.minimum(1,self.cap/np.maximum(norm,1e-12))
        limited=bool(np.any(scale<1-1e-6));position=combined*scale-lookahead
        at_cap=(bool(np.any(np.linalg.norm(position[0],axis=-1)>=self.cap-1e-4)) if self.cap_guard=='residual' else limited)
        if at_cap:
            if self.cap_since is None:self.cap_since=now_ns
        else:self.cap_since=None
        reasons=[]
        if not (np.isfinite(position).all() and np.isfinite(rotation).all()):reasons.append('nan')
        if self.cap_since is not None and (now_ns-self.cap_since)*1e-9>=self.cap_hold_s:reasons.append('sustained_cap')
        if np.max(np.linalg.norm(position[0]-self.pos[0],axis=-1))>self.max_step:reasons.append('step_jump')
        angular=np.max(np.linalg.norm(qlog(qmul(rotation[0],qinv(self.rot[0]))),axis=-1))
        if angular>min(self.angular_step,self.angular_rate*max(dt,0)):reasons.append('angular_jump')
        if force_fallback and np.linalg.norm(self.pos)>1e-8:reasons.append('reference_reset')
        if expired:reasons.append('expired')
        if reasons and self.fallback_start is None:
            self.fallback_start=now_ns;self.fallback_pos=self.pos.copy();self.fallback_rot=self.rot.copy()
            self.fallback_reason=reasons
            for reason in reasons:self.counters[reason]+=1
        falling=self.fallback_start is not None
        if falling:
            alpha=np.clip((now_ns-self.fallback_start)*1e-9/max(self.fade_s,1e-9),0,1)
            position=self.fallback_pos*(1-alpha);rotation=qblend(self.fallback_rot,neutral,alpha)
            self.cap_since=None
            if alpha>=1:
                self.fallback_start=None;self.previous_curve=None
                if 'nan' in self.fallback_reason or 'expired' in self.fallback_reason:self.curve=None
                self.freeze_pos=np.zeros_like(position);self.freeze_rot=neutral.copy();self.release_start=now_ns
        combined=lookahead+position;norm=np.linalg.norm(combined,axis=-1,keepdims=True)
        position=combined*np.minimum(1,self.cap/np.maximum(norm,1e-12))-lookahead
        self.pos=position.copy();self.rot=rotation.copy()
        output=np.concatenate((raw[...,:3]+lookahead+position,qmul(rotation,raw[...,3:7])),-1)
        return output.astype(np.float32),dict(dt_s=dt,fallback=falling,reasons=list(self.fallback_reason) if falling else [],
            cap_limited=limited,residual_position=position.copy(),residual_quaternion=rotation.copy(),
            angular_step_deg=float(np.rad2deg(angular)),counters=dict(self.counters))
