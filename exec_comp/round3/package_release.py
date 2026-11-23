"""A1-6 item 1: the round-3 release, still disabled by default.

The TorchScript network is unchanged from round 2 -- A1's weights and its
``correct_kp`` interface are not what round 3 changed. What the release adds is
a manifest that states the contract the round-3 evaluation actually validated
(chunk rate, call mode, grasp freeze, shared cap, mutual exclusion, fallback)
and a latency measurement of the complete host-side stage, not only the
network forward.
"""
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
from exec_comp.common import ROOT,CHUNK,CHUNK_FPS,SOURCE_CHECKPOINT_HASH,digest,save_json,log
from exec_comp.export_a1 import ExportableA1,fixture,measure
from exec_comp.round3.a1_executor import A1Executor,ExecutorLimits,MODES
from exec_comp.train_common import load_model

RELEASE=ROOT/"releases/a1_round3"


def stage_latency(executor,device,repeats=200,warmup=50):
    """Wall time of one complete batch-1 stage call, A1 forward included."""
    history={"q":torch.zeros(1,10,29,device=device),"dq":torch.zeros(1,10,29,device=device),
             "kp":torch.zeros(1,10,4,9,device=device),
             "root_pos":torch.zeros(1,3,device=device),
             "root_quat":torch.tensor([[1.,0.,0.,0.]],device=device)}
    history["kp"][...,3]=1.;history["kp"][...,8]=1.
    pose=torch.zeros(1,6,4,9,device=device);pose[...,3]=1.;pose[...,8]=1.
    window=torch.zeros(1,CHUNK,4,9,device=device);window[...,3]=1.;window[...,8]=1.
    arguments=dict(deployed_pose=pose,raw_pose=pose,
                   lookahead_offset=torch.zeros(1,6,4,3,device=device),
                   state_hist=history,window=window,
                   switched=torch.ones(1,dtype=torch.bool,device=device),
                   grasp_phase=torch.zeros(1,dtype=torch.bool,device=device))
    for _ in range(warmup):executor(**arguments)
    torch.cuda.synchronize();times=[]
    for _ in range(repeats):
        start=time.perf_counter();executor(**arguments);torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    return {"mean_ms":float(np.mean(times)),"median_ms":float(np.median(times)),
            "p95_ms":float(np.percentile(times,95)),"batch":1,"warmup":warmup,
            "measured_calls":repeats,
            "timing":"host wall clock + CUDA synchronization; GPU-resident resampled window/history; "
                     "network/crossfade/freeze/ramp/cap/fallback included; forces refresh each call (event-mode upper bound); excludes upstream CPU reference sampler, FK/history collection and disk logging"}


def host_latency(artifact,mode,history,goal,contract,device):
    """Additional host pipeline, including real CPU sampling and FK/history."""
    from exec_comp.round3.deployment_adapter import Round3Adapter
    from exec_comp.round3.switching import DeployedSwitching,make_chunk,quiet_deployment_logging
    from exec_comp.round3.metrics import canonical,to_links
    from exec_comp.diff.chain import sample_pose
    quiet_deployment_logging()
    adapter=Round3Adapter({'enable':True,'model':str(artifact),'mode':mode},device=device)
    for i in range(10):adapter.observe(history['q'][i],history['dq'][i],history['root_pos'][i],history['root_quat'][i])
    cfg={k:v['value'] for k,v in contract['switching_parameters'].items()}
    sw=DeployedSwitching(config=cfg,future_offsets_s=np.arange(6)*.02)
    native=sample_pose(goal[None],torch.arange(100,device=device)*2)[0]
    p,q=to_links(native);mp,mq=to_links(history['kp'][-1]);sw.publish_state(0,mp.cpu().numpy(),mq.cpu().numpy())
    sw.offer(make_chunk(0,0,0,15,(p.cpu().numpy(),q.cpu().numpy())),0)
    times=[]
    for i in range(120):
        now=i*20_000_000;deployed=sw.sample(now)  # Existing baseline sampler is outside the additional stage.
        torch.cuda.synchronize();start=time.perf_counter()
        adapter.observe(history['q'][-1],history['dq'][-1],history['root_pos'][-1],history['root_quat'][-1])
        raw=sw.sample_raw(now,deployed=deployed)
        window=sw.sample_window(now,np.arange(100)/30)
        adapter.correct(torch.tensor(canonical(deployed)[None],device=device),torch.tensor(canonical(raw)[None],device=device),
                        torch.tensor(canonical(window)[None],device=device),switched=True,grasp_phase=False)
        torch.cuda.synchronize()
        if i>=20:times.append((time.perf_counter()-start)*1000)
    adapter.close()
    return {'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
            'batch':1,'warmup':20,'measured_calls':100,
            'timing':'offline saved-state fixture; host wall clock incl FK/history, original extra100@30Hz reference sampler, CPU/GPU conversion, released TorchScript and guards; excludes baseline6-token sampling and disk logs; forced refresh every call'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default=str(ROOT/"runs/models/inverse_smooth_r2_best.pt"))
    parser.add_argument("--cerebellum-artifact",default=None)
    parser.add_argument("--mode",choices=MODES,default="stepwise")
    parser.add_argument("--contract",default=str(ROOT/"configs/online_execution_contract.json"))
    parser.add_argument("--selection",default=str(ROOT/"runs/round3/stage1.json"))
    parser.add_argument("--benchmark-only",action="store_true")
    parser.add_argument('--refresh-metadata',action='store_true')
    arguments=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);device="cuda:0"
    RELEASE.mkdir(parents=True,exist_ok=True)
    artifact=RELEASE/"a1_correct_kp.pt"
    model=load_model(arguments.checkpoint,device).requires_grad_(False)
    history,goal=fixture(device)
    if not arguments.benchmark_only and not arguments.refresh_metadata:
        if artifact.exists():raise FileExistsError(artifact)
        wrapper=ExportableA1(model).eval()
        with torch.no_grad():
            script=torch.jit.trace_module(wrapper,{"forward":(history,goal),
                                                   "correct_kp":(history,goal)},
                                          strict=False,check_trace=True)
            torch.jit.save(torch.jit.freeze(script.eval(),preserved_attrs=["correct_kp"]),str(artifact))
    script=torch.jit.load(str(artifact),map_location=device).eval()
    with torch.no_grad():
        from exec_comp.geometry import from_anchor,to_anchor
        from exec_comp.runtime import correct_encoded,encode_history
        encoded,origin,quat=encode_history(history,device)
        expected=from_anchor(correct_encoded(model,encoded,to_anchor(goal,origin,quat)[None])[0],origin,quat)
        actual=script.correct_kp(history,goal)
        difference=float((expected-actual).abs().max())
        bound=float((actual[...,:3]-goal[...,:3]).norm(dim=-1).max())
    if difference>1e-5 or bound>.050001:raise AssertionError((difference,bound))
    limits=ExecutorLimits()
    from exec_comp.round3.deployment_adapter import ScriptPredictor
    executor=A1Executor(enable=True,batch=1,device=device,mode=arguments.mode,
                        checkpoint=artifact,predictor=ScriptPredictor(artifact,device),limits=limits)
    contract=json.loads(Path(arguments.contract).read_text())
    selection=json.loads(Path(arguments.selection).read_text()) if Path(arguments.selection).exists() else None
    latency={"torchscript_network":measure(script,history,goal),
             "full_stage":stage_latency(executor,device),
             'offline_host_stage':host_latency(artifact,arguments.mode,history,goal,contract,device)}
    hardware=torch.cuda.get_device_name()
    short="4080s" if "4080" in hardware else "4090" if "4090" in hardware else hardware.replace(" ","_")
    save_json(RELEASE/f"latency_{short}.json",
              {"gpu":hardware,"torch":torch.__version__,"mode":arguments.mode,**latency})
    if arguments.benchmark_only:
        log("a1_round3_benchmark",gpu=hardware,**{k:v["mean_ms"] for k,v in latency.items()});return
    previous=json.loads((RELEASE/'manifest.json').read_text()) if (RELEASE/'manifest.json').exists() else {}
    manifest={
        "release":"a1_round3","format":"TorchScript FP32",
        "interface":"correct_kp(Dict[str,Tensor] state_hist, Tensor[100,4,9] K_star_chunk) -> Tensor[100,4,9]",
        "history":10,"keypoint_order":["root","head","left_wrist","right_wrist"],
        "rotation":"x/z matrix columns","position_residual_bound_m":.05,
        "artifact_sha256":digest(artifact),
        "a1_training_checkpoint":str(Path(arguments.checkpoint).resolve()),
        "a1_checkpoint_sha256":digest(arguments.checkpoint),
        "frozen_cerebellum_raw_sha256":SOURCE_CHECKPOINT_HASH,
        "network_identical_to":"releases/a1_round2/a1_correct_kp.pt (same weights and interface)",
        "deployed_chunk":{"fps":contract["chunk_fps"],"frame_count":contract["chunk_frame_count"],
            "replan_interval_s_median":contract["replan_interval_s"]["p50"],
            "source":"measured from the recorded closed-loop runs, not assumed"},
        "a1_input_window":{"frames":CHUNK,"spacing_hz":CHUNK_FPS,
            "built_from":"the stitched command stream the deployed switcher produces, not a chunk",
            "padding":"frames past the playing chunk's end repeat its last frame; the host "
                      "counts how often that happens"},
        "call_mode":{"selected":arguments.mode,"available":list(MODES),
            "stepwise":"one call per 50 Hz control step, then the 20 ms residual low pass",
            "switch_event":"one call per accepted chunk switch, with a linear residual "
                           "crossfade over seam_crossfade_s triggered by the switch itself",
            "selected_by":None if selection is None else selection.get("selected"),
            "seam_crossfade_s":limits.seam_crossfade_s},
        "grasp_behaviour":{"freeze":True,
            "triggers":["FOCUS handoff window","hand-event guard boundary","stall wrist assist",
                        "the brain's own predicted close boundary"],
            "release_ramp_s":limits.grasp_release_ramp_s,
            "on_exit":f"linear ramp back to the live residual over {limits.grasp_release_ramp_s*1000:.0f} ms",
            "never":"the residual is never set to zero in a single control step"},
        "shared_cap":{"total_offset_cap_m":limits.total_offset_cap_m,
            "applies_to":"anti-stall lookahead offset plus A1 residual, measured against raw K*",
            "rule":"project the vector sum onto the5cm ball; opposing residual/lookahead can cancel"},
        'lookahead_policy':{'keep':True if selection is None else not selection['selected'].endswith('stall-off'),
                            'note':'when disabled, the independent post-switch stage strips the original lookahead; switching code is not altered'},
        "mutual_exclusion":{"wrist_compensation":"none",
            "enforced_in":"exec_comp/round3/deployment_env.py before the robot starts"},
        "safety_fallback":{"ramp_s":limits.fallback_ramp_s,
            "nan":"any non-finite residual","sustained_cap_s":limits.cap_hold_s,
            "cap_guard_reading":limits.cap_guard,
            "single_step_jump_cm":limits.max_step_cm,
            "recovery":"after the fade the stage rejoins the live residual through the same ramp",
            "logged":"each trigger is counted and the first step of each fade carries the counters"},
        "logging":"per control step: raw K*, corrected command, measured keypoints, switch event, "
                  "grasp phase, residual, total offset and fallback state",
        "residual_lowpass_50hz":limits.lowpass,"lowpass_delay_ms":20 if limits.lowpass else 0,
        "fingertips":"the host wrapper passes every fingertip field through unchanged",
        "enabled_by_default":False,"real_robot_run":False,
        'host_executor_sha256':digest(ROOT/'round3/a1_executor.py'),
        'host_hooks_sha256':digest(ROOT/'round3/deployment_hooks.py'),
        'host_adapter_sha256':digest(ROOT/'round3/deployment_adapter.py'),
        'acceptance_status':'pending A1-5 full evaluation; packaging is not robot approval',
        "eager_max_abs":difference,"max_position_residual_m":bound}
    if selection:
        from exec_comp.round3.metrics import acceptance,verdict
        selected=selection['selected'];checks={};complete=True
        for name,folder,label,raw_label in [
            ('motion300',ROOT/'runs/round3/closed_loop',selected,'raw_stall-'+selected.split('stall-')[1]),
            ('DR300',ROOT/'runs/round3/closed_loop',selected+'_dr','raw_dr'),
            ('brain10',ROOT/'runs/round3/brain_evaluation',selected,'raw_stall-'+selected.split('stall-')[1])]:
            path=folder/label/'metrics.json';raw_path=folder/raw_label/'metrics.json'
            if not path.exists() or not raw_path.exists():complete=False;checks[name]=None;continue
            test=json.loads(path.read_text());raw=json.loads(raw_path.read_text())
            checks[name]=verdict(acceptance(test,raw))
            full=(test.get('robot_episodes')==10 if name=='brain10' else test['metrics']['all']['clips']==300)
            if not full or test.get('truncated_smoke'):complete=False
        manifest['acceptance_gates']=checks
        manifest['acceptance_status']=('pending A1-5 full evaluation' if not complete else
            'passed_simulation_gates_only' if all(v is True for case in checks.values() for v in case.values())
            else 'failed_simulation_gates')
    if arguments.cerebellum_artifact:
        manifest["cerebellum_deployment_path"]=arguments.cerebellum_artifact
        manifest["cerebellum_deployment_sha256"]=digest(arguments.cerebellum_artifact)
    elif previous.get('cerebellum_deployment_sha256'):
        for key in ('cerebellum_deployment_path','cerebellum_deployment_sha256'):
            manifest[key]=previous[key]
    save_json(RELEASE/"manifest.json",manifest)
    log("a1_round3_release",artifact=str(artifact),mode=arguments.mode,
        **{k:v["mean_ms"] for k,v in latency.items()})


if __name__=="__main__":main()
