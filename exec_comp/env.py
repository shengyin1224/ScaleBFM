"""Read-only reuse of the existing simulator with an isolated, deterministic config."""
import torch
import json
from exec_comp.common import ROOT,TRACK,ACTIVE,HEAD_OFFSET,metadata,import_path
from exec_comp.geometry import qapply,qmul,qconj,q_to_6d,pose9,six_to_q


def neutral_phase(env,command_name="motion",token_count=6):
    out=torch.zeros(env.num_envs,token_count,2,device=env.device);out[...,1]=2
    return out


def make_env(num_envs,split,device,randomized=False,seed=42):
    import gymnasium as gym
    import scaletrack.tasks
    from isaaclab.managers import EventTermCfg,TerminationTermCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from scaletrack.tasks.tracking.mdp.commands import MotionCommand
    module=import_path("exec_comp_readonly_env_cfg",TRACK/"scripts/precision_teacher_student/env_cfg.py")
    cfg=module.PrecisionEnvCfg()
    cfg.seed=seed;cfg.scene.num_envs=num_envs;cfg.sim.device=device
    # Cloned environments have collision filtering. A compact origin grid
    # avoids 40m float32 world-coordinate cancellation in the 1e-5 obs check.
    cfg.scene.env_spacing=.5
    cfg.scene.filter_collisions=True
    cfg.commands.motion.class_type=MotionCommand
    cfg.commands.motion.playback_enabled=False
    cfg.commands.motion.enable_reset_disturbance=False
    cfg.commands.motion.motion_file=str(ROOT/"configs"/(split+".yaml"))
    cfg.commands.motion.test_motion_file=None
    cfg.commands.motion.mode_sampling_weights={n:float(n=="HAT-4") for n in cfg.commands.motion.mode_candidates}
    cfg.rewards.wrist_precision=None
    cfg.observations.teacher_policy=None;cfg.observations.teacher_policy_task=None
    for name in ("policy","policy_task","critic","critic_task"):
        group=getattr(cfg.observations,name)
        group.enable_corruption=False
        for term in vars(group).values():
            if hasattr(term,"noise"):term.noise=None
    cfg.observations.policy_task.focus_phase.func=neutral_phase
    cfg.observations.critic_task.focus_phase.func=neutral_phase
    for name in dir(cfg.events):
        if isinstance(getattr(cfg.events,name),EventTermCfg):setattr(cfg.events,name,None)
    if randomized:
        from exec_comp.randomization import configure
        configure(cfg)
    # Record the physical terminal state before a manual reset; no reset jumps
    # are ever mislabeled as body-model transitions.
    for name in dir(cfg.terminations):
        if isinstance(getattr(cfg.terminations,name),TerminationTermCfg):setattr(cfg.terminations,name,None)
    env=RslRlVecEnvWrapper(gym.make("G1-HAT4-F2O-Finetune",cfg=cfg))
    robot=env.unwrapped.scene["robot"]
    torch.testing.assert_close(robot.data.body_com_pos_b[0,0],torch.tensor([0.,0.,-.07605],device=device),atol=1e-6,rtol=1e-5)
    assert list(robot.joint_names)==metadata()["joint_names"]
    scale=env.unwrapped.action_manager.get_term("joint_pos")._scale
    torch.testing.assert_close(scale.reshape(-1,29)[0],torch.tensor(metadata()["action_scale"],device=device))
    for actual,key in ((robot.data.default_joint_pos[0],"default_dof_pos"),
                       (robot.data.joint_stiffness[0],"stiffness"),
                       (robot.data.joint_damping[0],"damping"),
                       (robot.data.joint_effort_limits[0],"torque_limit")):
        expected=torch.tensor(metadata()[key],device=device,dtype=actual.dtype)
        if randomized and key in ('stiffness','damping'):
            assert torch.all(actual>=expected*.89999) and torch.all(actual<=expected*1.10001),key
        else:torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    return env


class Replay:
    def __init__(self,env,policy,sanitize_prefix=False):
        self.env=env;self.robot=env.unwrapped.scene["robot"];self.policy=policy
        self.command=env.unwrapped.command_manager.get_term("motion")
        self.n=env.num_envs;self.device=env.device
        self.body_ids=[self.robot.body_names.index(n) for n in metadata()["selected_body_names"]]
        self.offset=torch.tensor(HEAD_OFFSET,device=self.device)
        self.hist={}
        self.prefix=json.loads((ROOT/"configs/prefix_audit.json").read_text())["clips"] if (ROOT/"configs/prefix_audit.json").exists() else {}
        self.sanitize_prefix=sanitize_prefix

    def capture(self):
        d=self.robot.data
        p=d.body_pos_w[:,self.body_ids];r=d.body_quat_w[:,self.body_ids]
        ap=p[:,ACTIVE].clone();aq=r[:,ACTIVE]
        ap[:,1]+=qapply(aq[:,1],self.offset.expand(self.n,-1))
        state=torch.cat((d.joint_pos,d.joint_vel,d.root_ang_vel_b,d.projected_gravity_b),-1)
        return {"s":state,"q":d.joint_pos,"dq":d.joint_vel,"w":d.root_ang_vel_b,
                "root_p":p[:,0],"root_q":r[:,0],"kp":pose9(ap,aq),
                "privileged":torch.cat((p[:,0,2:3],d.root_lin_vel_b),-1)}

    def reference(self,offsets=None):
        offsets=list(range(6)) if offsets is None else list(offsets)
        frames=torch.as_tensor(offsets,device=self.device,dtype=torch.float32)
        lower=frames.floor().long().tolist();upper=frames.ceil().long().tolist()
        weight=(frames-frames.floor())[None,:,None,None]
        p0=self.command.body_pos_w_future_manual(lower)
        r0=self.command.body_quat_w_future_manual(lower)
        if lower!=upper:
            p1=self.command.body_pos_w_future_manual(upper);r1=self.command.body_quat_w_future_manual(upper)
            r1=torch.where((r0*r1).sum(-1,keepdim=True)<0,-r1,r1)
            p=p0+weight*(p1-p0);r=torch.nn.functional.normalize(r0+weight*(r1-r0),dim=-1)
        else:p=p0;r=r0
        ap=p[:,:,ACTIVE].clone();ar=r[:,:,ACTIVE]
        ap[:,:,1]+=qapply(ar[:,:,1],self.offset.expand(self.n,len(offsets),3))
        return pose9(ap,ar)

    def reset(self,ids,motion_ids):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        self.command.motion_ids[ids.cpu()]=torch.as_tensor(motion_ids,device="cpu",dtype=torch.long)
        # Source finite differences at frame2 can still include the frame1
        # startup jump, even when their speed falls below the 3m/s audit gate.
        # Use a uniform 3-frame guard as well as the audited invalid prefix.
        starts=[max(3,self.prefix[self.command.motion_names_train[int(i)]]["start"]) if self.sanitize_prefix else 0 for i in motion_ids]
        self.command.time_steps[ids.cpu()]=torch.as_tensor(starts,dtype=torch.long)-1
        self.command.randomize_next_resampling=False
        self.command.is_evaluating=False
        self.env.unwrapped._reset_idx(ids)
        self.env.unwrapped.scene.write_data_to_sim()
        self.env.unwrapped.sim.forward()
        self.env.unwrapped.scene.update(dt=.02)
        state=self.capture()
        for key,value in (("q",state["q"]),("dq",state["dq"]),("root_q",state["root_q"]),("w",state["w"])):
            if key not in self.hist:self.hist[key]=value[:,None].repeat(1,3,1).clone()
            self.hist[key][ids]=value[ids,None].expand(-1,3,-1)
        if "action" not in self.hist:self.hist["action"]=torch.zeros(self.n,3,29,device=self.device)
        self.hist["action"][ids]=0
        return state

    def append(self,state,action):
        for key,value in (("q",state["q"]),("dq",state["dq"]),("root_q",state["root_q"]),("w",state["w"]),("action",action)):
            self.hist[key]=torch.cat((self.hist[key][:,1:],value[:,None]),1)

    def action(self,state,command_kp):
        n=self.n
        p=self.command.body_pos_w_future_manual([0,1,2,3,4,5]).clone()
        r=self.command.body_quat_w_future_manual([0,1,2,3,4,5]).clone()
        active_q=six_to_q(command_kp[...,3:])
        active_p=command_kp[...,:3].clone()
        active_p[:,:,1]-=qapply(active_q[:,:,1],self.offset.expand(n,6,3))
        p[:,:,ACTIVE]=active_p;r[:,:,ACTIVE]=active_q
        inverse=qconj(state["root_q"])[:,None,None].expand(n,6,14,4)
        target_p=qapply(inverse,p-state["root_p"][:,None,None])
        target_r=qmul(inverse,r)
        h=self.hist
        return self.policy(h["q"],h["dq"],h["root_q"],h["w"],h["action"],target_p,target_r)

    def consistency_check(self,state):
        from exec_comp.diff.obs import build_obs
        q=state["q"];local_p,local_r=self.policy.fk(q)
        indices=[self.robot.body_names.index(n) for n in self.policy.fk.names]
        actual=self.robot.data.body_pos_w[:,indices]
        expected=qapply(state["root_q"][:,None].expand(-1,len(indices),-1),local_p)+state["root_p"][:,None]
        maximum=float((actual-expected).abs().max())
        if maximum>=1e-3:raise AssertionError(f"Isaac FK mismatch {maximum} m")
        # Fill the environment's own history with the same initial state used
        # by the independent policy history. This check uses actual ObsManager
        # output, not a second copy of our observation formula.
        for _ in range(3):oracle=self.env.unwrapped.observation_manager.compute(update_history=True)
        ref_p=self.command.body_pos_w_future_manual(list(range(6)))
        ref_q=self.command.body_quat_w_future_manual(list(range(6)))
        inverse=qconj(state["root_q"])[:,None,None].expand(self.n,6,14,4)
        p=qapply(inverse,ref_p-state["root_p"][:,None,None]);r=qmul(inverse,ref_q)
        h=self.hist
        prop,task,action=build_obs(self.policy.fk,h["q"],h["dq"],h["root_q"],h["w"],h["action"],p,r)
        mapping=oracle["mode_mapping"]
        if mapping.ndim==2:mapping=mapping[:,None]
        mode=oracle["mode"]
        if mode.ndim==2:mode=mode[:,None].expand(-1,6,-1)
        oracle_task=torch.cat((oracle["policy_task"]*mapping,mode),-1)
        errors={"proprio":float((prop-oracle["policy"]).abs().max()),
                "task":float((task-oracle_task).abs().max()),
                "action_history":float((action-oracle["action"]).abs().max())}
        if max(errors.values())>=1e-5:raise AssertionError(f"Isaac observation mismatch: {errors}")
        return {"fk_max_m":maximum,"obs_max_abs":errors}
