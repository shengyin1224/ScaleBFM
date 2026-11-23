"""Headless MuJoCo only. Never imports deployment envs, DDS, cameras or viewers."""
from pathlib import Path
from collections import deque
import os
import numpy as np
import mujoco
from exec_comp.common import PROJECT,metadata
from exec_comp.round5.engine import rotate,qinv

XML=Path(os.environ.get('R6_INSPIRE_XML',str(PROJECT/'exec_comp/round6/assets/g1_29dof_inspire_hand.xml')))


class World:
    def __init__(self,xml=None,native_pd=True):
        self.model=mujoco.MjModel.from_xml_path(str(xml or XML));self.model.opt.timestep=.002
        self.native_pd=bool(native_pd)
        self.data=mujoco.MjData(self.model);self.observed_data=mujoco.MjData(self.model);m=self.model;meta=metadata()
        joints=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in meta['joint_names']]);assert (joints>=0).all()
        self.joints=joints;self.qa=m.jnt_qposadr[joints];self.va=m.jnt_dofadr[joints]
        self.ai=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in meta['action_names']]);assert (self.ai>=0).all()
        self.body_ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in meta['selected_body_names']]);assert (self.body_ids>=0).all()
        self.kp=self.body_ids[[0,7,10,13]]
        self.hand_joints=np.array([i for i in range(m.njnt) if i not in joints and m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE],int)
        self.hq=m.jnt_qposadr[self.hand_joints];self.hv=m.jnt_dofadr[self.hand_joints]
        self.nominal_mass=m.body_mass.copy();self.nominal_inertia=m.body_inertia.copy();self.nominal_friction=m.geom_friction.copy()
        self.distal=[]
        for name in ('left_wrist_yaw_link','right_wrist_yaw_link'):
            root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);ids=[]
            for i in range(m.nbody):
                j=i
                while j>0 and j!=root:j=int(m.body_parentid[j])
                if j==root:ids.append(i)
            self.distal.append(np.array(ids))
        self.base_kp=np.asarray(meta['stiffness']);self.base_kd=np.asarray(meta['damping']);self.limit=np.asarray(meta['torque_limit'])
        self.hand_ai=np.array([i for i in range(m.nu) if i not in self.ai],int)
        self.original_ctrlrange=m.actuator_ctrlrange.copy()
        if self.native_pd:
            if not np.allclose(m.actuator_gear[:,0],1.) or not np.allclose(m.actuator_gear[:,1:],0.):raise ValueError('Native PD requires unit hinge motor gear')
            # Same generalized PD forces as the Python substep loop, evaluated
            # in MuJoCo C++ at500Hz. Keep XML mass/inertia/contact parameters.
            m.actuator_biastype[:]=mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_ctrllimited[:]=False;m.actuator_forcelimited[:]=True
            m.actuator_forcerange[self.ai,0]=np.maximum(-self.limit,self.original_ctrlrange[self.ai,0])
            m.actuator_forcerange[self.ai,1]=np.minimum(self.limit,self.original_ctrlrange[self.ai,1])
            m.actuator_forcerange[self.hand_ai]=[-.8,.8]
        self.reset(np.asarray(meta['default_dof_pos']),np.array([0.,0.,.793]),np.array([1.,0.,0.,0.]))

    def reset(self,q,p,quat,dq=None,v=None,w=None,rng=None):
        m,d=self.model,self.data;m.body_mass[:]=self.nominal_mass;m.body_inertia[:]=self.nominal_inertia;m.geom_friction[:]=self.nominal_friction
        self.kp_gain=self.base_kp.copy();self.kd_gain=self.base_kd.copy();self.delay_s=0.;self.noise=False
        self.domain=dict(hand_load_scale=[1.,1.],pd_kp_scale=1.,pd_kd_scale=1.,delay_ms=0.,friction=float(m.geom_friction[0,0]))
        if rng is not None:
            scales=rng.uniform(.7,1.3,2)
            for ids,scale in zip(self.distal,scales):m.body_mass[ids]*=scale;m.body_inertia[ids]*=scale
            pk=rng.uniform(.7,1.3);dk=rng.uniform(.7,1.3);self.kp_gain*=pk;self.kd_gain*=dk
            self.delay_s=float(rng.uniform(0,.06));mu=float(rng.uniform(.3,1.3));m.geom_friction[:,0]=mu;self.noise=True
            self.domain=dict(hand_load_scale=scales.tolist(),pd_kp_scale=pk,pd_kd_scale=dk,delay_ms=self.delay_s*1000,friction=mu)
        if self.native_pd:
            m.actuator_gainprm[self.ai,0]=self.kp_gain
            m.actuator_biasprm[self.ai,1]=-self.kp_gain;m.actuator_biasprm[self.ai,2]=-self.kd_gain
            m.actuator_gainprm[self.hand_ai,0]=5.;m.actuator_biasprm[self.hand_ai,1]=-5.;m.actuator_biasprm[self.hand_ai,2]=-.1
        mujoco.mj_setConst(m,d);mujoco.mj_resetData(m,d)
        d.qpos[:3]=p;d.qpos[3:7]=quat;d.qpos[self.qa]=q
        if dq is not None:d.qvel[self.va]=dq
        if v is not None:d.qvel[:3]=v
        if w is not None:d.qvel[3:6]=w
        d.ctrl[:]=0.;mujoco.mj_forward(m,d);self.queue=deque();self.last_target=np.asarray(q).copy();self.hand_target=np.zeros(len(self.hq))
        self.rng=rng or np.random.default_rng(0)
        return self.capture()

    def capture(self):
        m,d=self.model,self.data
        # mj_step integrates qpos after its forward-kinematics pass. Refresh
        # before labelling this new state; otherwise xpos lags qpos by2ms.
        mujoco.mj_kinematics(m,d)
        p=d.qpos[:3].copy();quat=d.qpos[3:7].copy();q=d.qpos[self.qa].copy();dq=d.qvel[self.va].copy();w=d.qvel[3:6].copy()
        gravity=rotate(qinv(quat),np.array([0.,0.,-1.]));s=np.r_[q,dq,w,gravity]
        comvel=rotate(qinv(quat),d.qvel[:3])+np.cross(w,m.body_ipos[self.body_ids[0]])
        positions=d.xpos[self.kp].copy();rot=d.xquat[self.kp].copy();positions[1]+=rotate(rot[1],np.array([0.,0.,.4]))
        x=rotate(rot,np.broadcast_to(np.array([1.,0,0]),(4,3)));z=rotate(rot,np.broadcast_to(np.array([0.,0,1.]),(4,3)))
        return dict(q=q,dq=dq,w=w,p=p,quat=quat,s=s,privileged=np.r_[p[2],comvel],kp=np.concatenate((positions,x,z),-1))

    def step(self,target,focus=False):
        m,d=self.model,self.data
        self.queue.append((d.time+self.delay_s,np.asarray(target).copy()))
        # Passive finger actuation is replaced by a bilateral PD force, because
        # XML hand motor ctrlrange is one-sided and labelled in degrees rather
        # than torque. Body actuators, XML masses/inertias/geometry remain intact.
        if len(self.hq):
            ranges=m.jnt_range[self.hand_joints]
            hand=np.clip(np.zeros(len(ranges)),ranges[:,0],ranges[:,1])
            if focus:hand=np.clip(ranges[:,0]+.6*(ranges[:,1]-ranges[:,0]),ranges[:,0],ranges[:,1])
            self.hand_target+=np.clip(hand-self.hand_target,-.04,.04)
        if self.native_pd:
            # Hand actuator order differs from joint order in the XML.
            qtargets=dict(zip(self.hand_joints,self.hand_target))
            d.ctrl[self.hand_ai]=[qtargets[int(m.actuator_trnid[i,0])] for i in self.hand_ai]
            left=10
            while left:
                while self.queue and self.queue[0][0]<=d.time+1e-10:self.last_target=self.queue.popleft()[1]
                d.ctrl[self.ai]=self.last_target
                count=left
                if self.queue:count=min(left,max(1,int(np.ceil((self.queue[0][0]-d.time-1e-10)/m.opt.timestep))))
                mujoco.mj_step(m,d,nstep=count);left-=count
            return self.capture()
        for _ in range(10):
            while self.queue and self.queue[0][0]<=d.time+1e-10:self.last_target=self.queue.popleft()[1]
            tau=self.kp_gain*(self.last_target-d.qpos[self.qa])-self.kd_gain*d.qvel[self.va]
            d.ctrl[self.ai]=np.clip(tau,-self.limit,self.limit)
            if len(self.hq):d.qfrc_applied[self.hv]=np.clip(5*(self.hand_target-d.qpos[self.hq])-.1*d.qvel[self.hv],-.8,.8)
            mujoco.mj_step(m,d)
        return self.capture()

    def observe(self,state=None):
        state=self.capture() if state is None else state
        if not self.noise:return {k:v.copy() for k,v in state.items()}
        from exec_comp.round5.engine import qmul,qexp
        s={k:v.copy() for k,v in state.items()};rng=self.rng
        s['q']+=rng.uniform(-.01,.01,29);s['dq']+=rng.uniform(-.5,.5,29);s['w']+=rng.uniform(-.2,.2,3)
        s['p']+=rng.uniform(-.005,.005,3);s['quat']=qmul(qexp(rng.uniform(-np.pi/180,np.pi/180,3)),s['quat'])
        s['s']=np.r_[s['q'],s['dq'],s['w'],rotate(qinv(s['quat']),np.array([0.,0.,-1.]))]
        d=self.observed_data;d.qpos[:]=self.data.qpos;d.qpos[:3]=s['p'];d.qpos[3:7]=s['quat'];d.qpos[self.qa]=s['q'];mujoco.mj_kinematics(self.model,d)
        p=d.xpos[self.kp].copy();q=d.xquat[self.kp].copy();p[1]+=rotate(q[1],np.array([0.,0.,.4]))
        s['kp']=np.concatenate((p,rotate(q,np.broadcast_to(np.array([1.,0.,0.]),p.shape)),rotate(q,np.broadcast_to(np.array([0.,0.,1.]),p.shape))),-1)
        return s

    def fallen(self):
        s=self.capture();up=rotate(s['quat'],np.array([0.,0.,1.]))
        return not np.isfinite(self.data.qpos).all() or s['p'][2]<.45 or up[2]<.3
