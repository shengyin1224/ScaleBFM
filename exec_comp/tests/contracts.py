"""Numerical checks against MuJoCo and the exact deployed TensorRT artifact."""
import argparse
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import ROOT,TRACK,PROJECT,metadata,log,save_json
from exec_comp.policy import FrozenPolicy
from exec_comp.diff.obs import build_obs
from exec_comp.geometry import qapply,qconj,qmul,q_to_6d,qnormalize


def main():
    p=argparse.ArgumentParser();p.add_argument("--trt",action="store_true");args=p.parse_args()
    import mujoco
    torch.set_num_threads(2);torch.manual_seed(42)
    device="cuda:0" if args.trt else "cpu"
    policy=FrozenPolicy().to(device)
    meta=metadata();fk=policy.fk
    xml=TRACK/"source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in meta["joint_names"]]
    qadr=np.array([model.jnt_qposadr[i] for i in joint_ids])
    body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in fk.names]
    maxima={"fk_position_m":0.,"fk_quaternion_sign_invariant":0.,"obs_max_abs":0.}
    if args.trt:
        import torch_tensorrt
        release=PROJECT/"MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1"
        trt=torch.jit.load(str(release/"model_20000_scalebridge_tensorrt_4080s.pt")).to(device).eval()
        jit=torch.jit.load(str(release/"model_20000_scalebridge_torchscript.pt")).to(device).eval()
        maxima.update({"trt_action_max_abs":0.,"torchscript_action_max_abs":0.})
    for _ in range(32):
        q=policy.default[None]+torch.randn(1,29,device=device)*.06
        quat=qnormalize(torch.tensor([[1.,0.,0.,0.]],device=device)+torch.randn(1,4,device=device)*.04)
        origin=torch.tensor([[.2,-.4,.8]],device=device)
        data.qpos[:3]=origin[0].cpu();data.qpos[3:7]=quat[0].cpu();data.qpos[qadr]=q[0].cpu();mujoco.mj_forward(model,data)
        local_p,local_r=fk(q)
        world_p=qapply(quat[:,None].expand(-1,len(fk.names),-1),local_p)+origin[:,None]
        world_r=qmul(quat[:,None].expand(-1,len(fk.names),-1),local_r)
        oracle_p=torch.tensor(data.xpos[body_ids],device=device,dtype=torch.float32)[None]
        oracle_r=torch.tensor(data.xquat[body_ids],device=device,dtype=torch.float32)[None]
        maxima["fk_position_m"]=max(maxima["fk_position_m"],float((world_p-oracle_p).abs().max()))
        maxima["fk_quaternion_sign_invariant"]=max(maxima["fk_quaternion_sign_invariant"],float(torch.minimum((world_r-oracle_r).abs().amax(-1),(world_r+oracle_r).abs().amax(-1)).max()))
        qh=q[:,None].repeat(1,3,1);dh=torch.randn_like(qh)*.1
        rh=quat[:,None].repeat(1,3,1);wh=torch.randn(1,3,3,device=device)*.1;ah=torch.randn_like(qh)*.1
        target_p=local_p[:,None,fk.selected].repeat(1,6,1,1)+torch.randn(1,6,14,3,device=device)*.02
        target_r=local_r[:,None,fk.selected].repeat(1,6,1,1)
        prop,task,_=build_obs(fk,qh,dh,rh,wh,ah,target_p,target_r)
        # Independent environment/MuJoCo construction in world coordinates.
        selected=oracle_p[:,fk.selected];selected_r=oracle_r[:,fk.selected]
        reference=qapply(quat[:,None,None].expand(1,6,14,4),target_p)+origin[:,None,None]
        reference_r=qmul(quat[:,None,None].expand_as(target_r),target_r)
        inverse=qconj(quat[:,None,None]).expand_as(target_r)
        p_to_root=qapply(inverse,reference-origin[:,None,None])
        p_error=qapply(inverse,reference-selected[:,None])
        r_to_root=qmul(inverse,reference_r)
        r_error=qmul(qmul(inverse,qmul(reference_r,qconj(selected_r[:,None]).expand_as(reference_r))),quat[:,None,None].expand_as(reference_r))
        raw=torch.cat((p_to_root.flatten(2),p_error.flatten(2),q_to_6d(r_to_root).flatten(2),q_to_6d(r_error).flatten(2),
                       torch.arange(6,device=device).view(1,6,1),torch.tensor([0.,2.],device=device).view(1,1,2).expand(1,6,2)),-1)
        mode=torch.zeros(1,14,device=device);mode[:,[0,7,10,13]]=1
        mask=torch.cat([mode.repeat_interleave(d,dim=-1) for d in (3,3,6,6)]+[torch.ones(1,3,device=device)],-1)
        oracle=torch.cat((raw*mask[:,None],mode[:,None].expand(-1,6,-1)),-1)
        maxima["obs_max_abs"]=max(maxima["obs_max_abs"],float((task-oracle).abs().max()))
        if args.trt:
            phase=torch.tensor([0.,2.],device=device).view(1,1,2).expand(1,6,2)
            inputs=(rh,wh,qh,dh,ah,target_p,target_r,torch.tensor([8],device=device),torch.arange(6,device=device).view(1,6,1),phase)
            with torch.no_grad():
                ours=policy(qh,dh,rh,wh,ah,target_p,target_r)
                maxima["trt_action_max_abs"]=max(maxima["trt_action_max_abs"],float((ours-trt(*inputs)[1]).abs().max()))
                maxima["torchscript_action_max_abs"]=max(maxima["torchscript_action_max_abs"],float((ours-jit(*inputs)[1]).abs().max()))
    target_p=target_p.detach().requires_grad_()
    action=policy(qh,dh,rh,wh,ah,target_p,target_r)
    gradient=torch.autograd.grad(action.square().sum(),target_p)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert maxima["fk_position_m"]<.001,maxima
    assert maxima["obs_max_abs"]<1e-5,maxima
    if args.trt:
        assert maxima["torchscript_action_max_abs"]<1e-5,maxima
        assert maxima["trt_action_max_abs"]<1e-4,maxima
    maxima["input_gradient_finite_nonzero"]=True
    log("contract_checks",**maxima)
    save_json(ROOT/"configs"/("deployment_contract_checks.json" if args.trt else "fk_obs_checks.json"),maxima)


if __name__=="__main__":main()
