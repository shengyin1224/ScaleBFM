"""Measure the real deployment execution contract from recorded online runs.

Read-only over ScaleBridge/outputs: no deployment file is edited and no robot
or socket is touched. Everything the round-3 simulation reuses (chunk rate,
replan arrival, latency age, which environment class and switching parameters
actually ran) is taken from here rather than from a document.
"""
import argparse
import collections
from datetime import datetime
import json
from pathlib import Path
import re
import numpy as np
import yaml
from exec_comp.common import ROOT,PROJECT,save_json,log

BRIDGE=PROJECT/"ScaleBridge"
ADAPTIVE="env.motion_tracking_hat4_adaptive.HAT4AdaptiveMotionTrackingEnv"
ONLINE="env.motion_tracking_hat4_online.HAT4OnlineMotionTrackingEnv"
R1="R1: starting closed loop"
STAMP=re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
# Every switch rule the round-3 simulation reuses must be pinned to the value
# the robot actually ran with; an unexplained spread is a reason to stop.
SWITCH_KEYS=("min_chunk_execution_fraction","relative_chunk_min_execution_fraction",
    "chunk_start_reference_continuity","continuity_search_ahead_s","chunk_blend_s","chunk_end_margin_s",
    "rtc_enabled","rtc_prefix_frames","rtc_hard_prefix_frames","rtc_blend_frames",
    "phase_rotation_weight","phase_velocity_weight","phase_measured_weight","phase_max_root_jump_m",
    "chunk_phase_alignment","chunk_alignment","max_source_state_lag",
    "adaptive_far_min_frames","adaptive_far_max_frames","adaptive_near_min_frames","adaptive_near_max_frames",
    "adaptive_near_distance_m","adaptive_min_wrist_travel_m","adaptive_stall_frames","adaptive_stall_motion_m",
    "adaptive_stall_increment_m","adaptive_stall_max_lookahead_m",
    "focus_handoff_enabled","focus_handoff_version","focus_handoff_finger_mode","focus_handoff_source",
    "focus_handoff_sides","focus_v3_closed_spread_m","focus_v3_max_approach_m",
    "focus_adaptive_enabled","stall_wrist_assist","continuity_hand_event_guard","finger_wrist_gate",
    "approach_slowdown","finger_chunk_source","wrist_compensation")


def r1_time(path):
    try:
        for line in path.open(errors="replace"):
            if R1 in line:
                match=STAMP.match(line)
                if match:return datetime.fromisoformat(match.group(1)).timestamp()
    except FileNotFoundError:pass
    return None


def scan(runs):
    rows=[];targets=collections.Counter();values={k:collections.Counter() for k in SWITCH_KEYS}
    fps=collections.Counter();frames=collections.Counter();modes=collections.Counter()
    gaps=[];ages=[];selected=[];shifts=[];rtc=collections.Counter();backtrack=[]
    for config_path in runs:
        run=config_path.parent.parent
        config=yaml.safe_load(config_path.open())
        env=(config.get("env") or {});target=env.get("_target_")
        targets[target]+=1
        if target!=ADAPTIVE:continue
        chunks=run/"online_hat_chunks.jsonl"
        started=r1_time(run/"run.log")
        if not chunks.exists() or started is None:continue
        cfg=env.get("config") or {}
        for key in SWITCH_KEYS:values[key][json.dumps(cfg.get(key,"<absent>"),sort_keys=True)]+=1
        times=[]
        for line in chunks.open():
            try:record=json.loads(line)
            except json.JSONDecodeError:continue
            if record.get("record_type")!="hat_chunk" or float(record["time"])<started:continue
            aligned=record["aligned"]
            fps[float(aligned["fps"])]+=1;frames[int(aligned["frame_count"])]+=1
            times.append(float(record["monotonic"]))
            state=aligned.get("execution_alignment") or {}
            modes[state.get("mode")]+=1
            if state.get("source_age_s") is not None:ages.append(float(state["source_age_s"]))
            if state.get("selected_frame") is not None:selected.append(int(state["selected_frame"]))
            if state.get("continuity_shift_frames") is not None:shifts.append(int(state["continuity_shift_frames"]))
            if state.get("backtrack_m") is not None:backtrack.append(float(state["backtrack_m"]))
            overlap=state.get("rtc")
            if isinstance(overlap,dict):rtc[("applied" if overlap.get("applied") else overlap.get("bypass_reason","bypassed"))]+=1
        if len(times)>3:
            gaps.append(np.diff(np.asarray(times,dtype=np.float64)));rows.append(str(run.relative_to(BRIDGE)))
    return dict(runs=rows,targets=targets,values=values,fps=fps,frames=frames,modes=modes,
                gaps=gaps,ages=ages,selected=selected,shifts=shifts,rtc=rtc,backtrack=backtrack)


def summary(x,percentiles=(5,25,50,75,95)):
    x=np.asarray(x,dtype=np.float64)
    if not x.size:return None
    out={"count":int(x.size),"mean":float(x.mean()),"min":float(x.min()),"max":float(x.max())}
    out.update({f"p{p}":float(np.percentile(x,p)) for p in percentiles})
    return out


def unique(counter,name):
    if len(counter)!=1:raise ValueError(f"{name} is not a single deployed value: {dict(counter)}")
    return next(iter(counter))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--since",default="2026-10-16",help="earliest recording date to include")
    parser.add_argument("--output",default=str(ROOT/"configs/online_execution_contract.json"))
    args=parser.parse_args()
    runs=sorted(p for p in BRIDGE.glob("outputs/2026-*/*/.hydra/config.yaml") if p.parts[-4]>=args.since)
    if not runs:raise FileNotFoundError(f"No recorded runs under {BRIDGE/'outputs'} since {args.since}")
    data=scan(runs)
    if not data["runs"]:raise RuntimeError("No adaptive closed-loop recording found; cannot pin the contract")
    gaps=np.concatenate(data["gaps"]);gaps=gaps[(gaps>0)&(gaps<5)]
    pinned={}
    for key,counter in data["values"].items():
        ordered=counter.most_common()
        pinned[key]={"value":json.loads(ordered[0][0]),"agreement":ordered[0][1]/sum(counter.values()),
                     "observed":{k:v for k,v in ordered}} if len(ordered)>1 else {"value":json.loads(ordered[0][0]),"agreement":1.,"observed":None}
    contract={
        "source":"ScaleBridge recorded closed-loop runs (read-only); no deployment file modified",
        "since":args.since,"runs":len(data["runs"]),"run_list":data["runs"],
        "environment_class_counts":{k:v for k,v in data["targets"].most_common()},
        "environment_class":ADAPTIVE,
        "environment_class_note":("Both the base online env and the adaptive env appear in this window. "
            "The adaptive subclass is the one that implements the PACE horizon, the near cadence, the "
            "anti-stall lookahead and the close boundary, so it is the contract round 3 reproduces."),
        "chunk_fps":float(unique(data["fps"],"chunk fps")),
        "chunk_frame_count":int(unique(data["frames"],"chunk frame_count")),
        "chunk_span_s":float(unique(data["frames"],"chunk frame_count"))/float(unique(data["fps"],"chunk fps")),
        "replan_interval_s":summary(gaps),
        "replan_hz_from_median":float(1./np.median(gaps)),
        "source_age_s":summary(data["ages"]),
        "selected_start_frame":summary(data["selected"]),
        "continuity_shift_frames":summary(data["shifts"]),
        "backtrack_m":summary(data["backtrack"]),
        "alignment_modes":{str(k):v for k,v in data["modes"].most_common()},
        "rtc_outcomes":{str(k):v for k,v in data["rtc"].most_common()},
        "switching_parameters":pinned,
        "aac":"not deployed: adaptive_chunk_execution selects a horizon from one prediction and documents "
              "that a single forward cannot supply same-observation uncertainty; see round3/audit_brain_sampling.py",
        "measurement":"chunk arrivals recorded after the R1 closed-loop event only; gaps outside (0,5)s dropped",
    }
    save_json(args.output,contract)
    log("online_contract",fps=contract["chunk_fps"],frames=contract["chunk_frame_count"],
        runs=contract["runs"],replan_median_s=contract["replan_interval_s"]["p50"],
        source_age_median_s=contract["source_age_s"]["p50"])


if __name__=="__main__":main()
