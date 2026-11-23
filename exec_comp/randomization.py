"""Isolated round-two robustness configuration; no existing deployment edits."""
import torch
from exec_comp.common import metadata
from exec_comp.geometry import from_anchor,six_to_q,qconj,qmul,q_to_6d


def material_uniform(env,env_ids,asset_cfg):
    asset=env.scene[asset_cfg.name]
    material=asset.root_physx_view.get_material_properties()
    coeff=torch.rand(material.shape[:2])*.9+.3
    material[...,0]=coeff;material[...,1]=coeff;material[...,2]=0
    asset.root_physx_view.set_material_properties(material,torch.arange(material.shape[0],dtype=torch.int32))


def configure(cfg):
    from isaaclab.managers import EventTermCfg,SceneEntityCfg
    from isaaclab.envs import mdp
    import scaletrack.tasks.tracking.mdp as tracking_mdp
    cfg.scene.terrain.physics_material.friction_combine_mode='multiply'
    cfg.scene.terrain.physics_material.static_friction=1.
    cfg.scene.terrain.physics_material.dynamic_friction=1.
    cfg.events.exec_comp_material=EventTermCfg(func=material_uniform,mode="startup",params={"asset_cfg":SceneEntityCfg("robot")})
    cfg.events.exec_comp_gains=EventTermCfg(func=mdp.randomize_actuator_gains,mode="startup",params={
        "asset_cfg":SceneEntityCfg("robot",joint_names=".*"),"stiffness_distribution_params":(.9,1.1),
        "damping_distribution_params":(.9,1.1),"operation":"scale","distribution":"uniform"})
    cfg.events.exec_comp_torso_com=EventTermCfg(func=tracking_mdp.randomize_rigid_body_com,mode="startup",params={
        "asset_cfg":SceneEntityCfg("robot",body_names="torso_link"),"com_range":{"x":(-.025,.025),"y":(-.05,.05),"z":(-.05,.05)}})


class PolicyNoise:
    def __init__(self,policy,device,sensor=None,seed=202610251):
        self.original=policy.from_obs;self.generator=torch.Generator(device=device).manual_seed(seed)
        self.history=None;self.sensor=sensor;self.fk=getattr(policy,'fk',None);self.default=getattr(policy,'default',None)

    def uniform(self,like):return torch.rand(like.shape,dtype=like.dtype,device=like.device,generator=self.generator)*2-1

    def __call__(self,prop,task,action_hist):
        scale=prop.new_tensor([.05]*3+[.2]*3+[.01]*29+[.025]*29)
        current=self.uniform(prop[:,-1])*scale
        if self.sensor is not None:
            current[:,6:35]=self.sensor.q_error;current[:,35:64]=self.sensor.dq_error*.05
            if hasattr(self.sensor,'imu_error'):current[:,:6]=self.sensor.imu_error
        if self.history is None:self.history=current[:,None].repeat(1,3,1)
        else:self.history=torch.cat((self.history[:,1:],current[:,None]),1)
        noisy_task=task.clone();mask=task[...,-14:]
        if self.fk is not None:
            noisy_q=prop[:,-1,6:35]+current[:,6:35]+self.default
            position,quat=self.fk(noisy_q)
            position=position[:,self.fk.selected];quat=quat[:,self.fk.selected]
            target_pos=task[...,:42].reshape(-1,6,14,3)
            target_quat=six_to_q(task[...,84:168].reshape(-1,6,14,6))
            noisy_task[...,42:84]=(target_pos-position[:,None]).flatten(2)*mask.repeat_interleave(3,-1)
            relative=qmul(target_quat,qconj(quat[:,None]).expand_as(target_quat))
            noisy_task[...,168:252]=q_to_6d(relative).flatten(2)*mask.repeat_interleave(6,-1)
        return self.original(prop+self.history,noisy_task,action_hist)


class StateNoise:
    """Joint sensor noise reaches A1's q/dq and its FK-derived measured keypoints."""
    def __init__(self,policy,device,seed=202610252):
        self.fk=policy.fk;self.seed=seed;self.generator=torch.Generator(device=device).manual_seed(seed)

    def __call__(self,state):
        result=dict(state)
        self.q_error=(torch.rand(state['q'].shape,device=state['q'].device,generator=self.generator)*2-1)*.01
        self.dq_error=(torch.rand(state['dq'].shape,device=state['dq'].device,generator=self.generator)*2-1)*.5
        q=state['q']+self.q_error;dq=state['dq']+self.dq_error
        result.update(q=q,dq=dq,kp=from_anchor(self.fk.keypoints(q),state['root_p'],state['root_q']))
        return result


class ActionDelay:
    def __init__(self,n,device,seed=202610253):
        self.seed=seed;generator=torch.Generator(device=device).manual_seed(seed)
        self.steps=torch.randint(0,3,(n,),device=device,generator=generator)
        self.history=None;self.reset_mask=torch.zeros(n,dtype=torch.bool,device=device)

    def reset(self,ids):self.reset_mask[ids]=True

    def __call__(self,action):
        if self.history is None:self.history=action[:,None].repeat(1,3,1)
        else:self.history=torch.cat((self.history[:,1:],action[:,None]),1)
        self.history=torch.where(self.reset_mask[:,None,None],action[:,None].expand(-1,3,-1),self.history)
        self.reset_mask[:]=False
        return self.history[torch.arange(len(action),device=action.device),2-self.steps]


def audit(env,delay):
    robot=env.unwrapped.scene['robot'];material=robot.root_physx_view.get_material_properties()
    kp=robot.data.joint_stiffness.clone();kd=robot.data.joint_damping.clone()
    for actuator in robot.actuators.values():
        kp[:,actuator.joint_indices]=actuator.stiffness
        kd[:,actuator.joint_indices]=actuator.damping
    for actual,key in ((kp,'stiffness'),(kd,'damping')):
        expected=actual.new_tensor(metadata()[key])
        assert torch.all(actual>=expected*.89999) and torch.all(actual<=expected*1.10001),key
    return {"friction_range":[.3,1.2],"pd_scale_range":[.9,1.1],"action_delay_control_steps":delay.steps.cpu().tolist(),
            "stiffness":kp.cpu().tolist(),"damping":kd.cpu().tolist(),
            "actuator_types":{name:type(actuator).__name__ for name,actuator in robot.actuators.items()},
            "ground_friction":1.,"ground_friction_combine_mode":"multiply",
            "torso_com_b":robot.data.body_com_pos_b[:,robot.body_names.index('torso_link')].cpu().tolist(),
            "material_friction_min":float(material[...,0].min()),"material_friction_max":float(material[...,0].max()),
            "observation_noise":"gravity±.05, gyro±.2rad/s, shared q±.01rad/dq±.5rad/s sensors for policy and A1; task FK/errors recomputed from the same noisy q. Software target commands have no artificial ±5cm noise. Root world tracker noise not additionally modeled.",
            "delay_start":"queue filled with first proposed action to avoid an artificial zero-action startup transient","seed":int(env.unwrapped.cfg.seed),'delay_seed':delay.seed}
