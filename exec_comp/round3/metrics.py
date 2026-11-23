"""Shared conversions and acceptance metrics for the round-3 evaluations."""
import numpy as np
import torch
from exec_comp.geometry import six_to_q

GATES={"actual_above_5hz_ratio":("<=",1.5),
       "action_difference_ratio":("<=",1.5),
       "joint_velocity_rms_ratio":("<=",1.5),
       "plan_seam_excess_p95_cm":("<=",.5),
       "wrist_error_reduction":(">=",.40)}


def to_links(poses):
    """pose9 [...,4,9] -> (positions, quaternions) as the chunk protocol wants."""
    return poses[...,:3],six_to_q(poses[...,3:])


def quat_columns(quat):
    """x and z columns of the rotation matrix, the cerebellum's 6-D convention."""
    w,x,y,z=(quat[...,i] for i in range(4))
    column_x=np.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),-1)
    column_z=np.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),-1)
    return column_x,column_z


def canonical(sample):
    """Deployment sample -> canonical [F,4,9] root/head/left/right keypoints."""
    positions=np.stack([sample[k] for k in ("root_pos","head_pos","left_wrist_pos","right_wrist_pos")],1)
    quaternions=np.stack([sample[k] for k in ("root_quat_wxyz","head_quat_wxyz",
                                              "left_wrist_quat_wxyz","right_wrist_quat_wxyz")],1)
    column_x,column_z=quat_columns(quaternions)
    return np.concatenate((positions,column_x,column_z),-1).astype(np.float32)


def distribution(values):
    values=np.asarray(values,dtype=np.float64)
    if not values.size:return None
    return {"count":int(values.size),"mean":float(values.mean()),
            "median":float(np.median(values)),"p95":float(np.percentile(values,95))}


def acceptance(corrected,baseline):
    """The A1-5 item 7 numbers, every one of them measured against raw."""
    def ratio(key):
        top=(corrected.get("jitter") or {}).get(key);bottom=(baseline.get("jitter") or {}).get(key)
        return None if top is None or not bottom or not np.isfinite(top) or not np.isfinite(bottom) else float(top/bottom)
    wrist=corrected["summary"]["wrist_mean_cm"];base=baseline["summary"]["wrist_mean_cm"]
    seam=corrected["summary"].get("plan_seam_p95_cm")
    reference_seam=baseline["summary"].get("plan_seam_p95_cm")
    return {"actual_above_5hz_ratio":ratio("actual_above_5hz_energy_fraction"),
            "action_difference_ratio":ratio("action_difference"),
            "joint_velocity_rms_ratio":ratio("joint_velocity_rms_rad_s"),
            "command_above_5hz_ratio":ratio("command_above_5hz_energy_fraction"),
            "plan_seam_p95_cm":seam,"raw_plan_seam_p95_cm":reference_seam,
            "plan_seam_excess_p95_cm":None if seam is None or reference_seam is None else seam-reference_seam,
            "wrist_mean_cm":wrist,"raw_wrist_mean_cm":base,
            "wrist_error_reduction":None if wrist is None or not base or not np.isfinite(wrist) or not np.isfinite(base) else float(1.-wrist/base)}


def verdict(values):
    """Pass/fail per gate, with None for a metric that could not be computed."""
    result={}
    for key,(sense,threshold) in GATES.items():
        value=values.get(key)
        if value is None or not np.isfinite(value):result[key]=None
        else:result[key]=bool(value<=threshold if sense=="<=" else value>=threshold)
    return result
