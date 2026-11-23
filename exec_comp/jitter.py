"""50Hz physical and command diagnostics; no concatenation across clip boundaries."""
import numpy as np
from scipy.signal import welch


def spectral_energy(position, fs=50.):
    x=np.asarray(position,dtype=np.float64)
    if len(x)<16:return {"high":0.,"total":0.,"frames":0}
    frequency,power=welch(x,fs=fs,axis=0,nperseg=min(256,len(x)),detrend="linear",window="hann")
    return {"high":float(power[frequency>5].sum()),"total":float(power.sum()),"frames":len(x)}


def clip_jitter(command,actual,action,dq,seams):
    c=np.asarray(command)[:,2:4,:3];a=np.asarray(actual)[:,2:4,:3]
    second=np.linalg.norm(np.diff(c,n=2,axis=0),axis=-1)*100
    diff=np.linalg.norm(np.diff(np.asarray(action),axis=0),axis=-1)
    velocity=np.asarray(dq,dtype=np.float64)
    seams=np.asarray(seams,dtype=np.float64)
    return {"frames":len(c),"command_second_difference_cm_sum":float(second.sum()),"command_second_difference_count":int(second.size),
            "command_spectrum":spectral_energy(c),"actual_spectrum":spectral_energy(a),
            "action_difference_sum":float(diff.sum()),"action_difference_count":len(diff),
            "joint_velocity_squared_sum":float((velocity**2).sum()),"joint_velocity_count":int(velocity.size),
            "seam_jump_cm":(np.linalg.norm(seams,axis=-1)*100).reshape(-1).tolist() if seams.size else []}


def aggregate_jitter(clips):
    def ratio(a,b):return sum(c[a] for c in clips)/max(1,sum(c[b] for c in clips))
    result={"command_second_difference_cm":ratio("command_second_difference_cm_sum","command_second_difference_count"),
            "action_difference":ratio("action_difference_sum","action_difference_count"),
            "joint_velocity_rms_rad_s":ratio("joint_velocity_squared_sum","joint_velocity_count")**.5}
    for name in ("command","actual"):
        high=sum(c[name+"_spectrum"]["high"]*c[name+"_spectrum"]["frames"] for c in clips)
        total=sum(c[name+"_spectrum"]["total"]*c[name+"_spectrum"]["frames"] for c in clips)
        result[name+"_above_5hz_energy_fraction"]=high/total if total>0 else None
    seams=np.asarray([v for c in clips for v in c["seam_jump_cm"]])
    result["chunk_seam_cm"]={"mean":float(seams.mean()),"median":float(np.median(seams)),"p95":float(np.percentile(seams,95))} if seams.size else None
    result["spectrum_definition"]="Welch Hann, detrend=linear, nperseg<=256, >5Hz/total, xyz+both wrists summed, frame-weighted across clips"
    return result


class ResidualLowPass:
    """Alpha=.5 at50Hz: 20ms low-frequency delay; filter only compensation."""
    def __init__(self):self.previous=None

    def __call__(self,command,raw):
        import torch
        from exec_comp.geometry import pose_normalize
        delta=command-raw
        previous=torch.zeros_like(delta[:,0]) if self.previous is None else self.previous
        result=[]
        for value in delta.unbind(1):
            previous=.5*value+.5*previous;result.append(previous)
        filtered=torch.stack(result,1)
        self.previous=filtered[:,0].detach()
        return pose_normalize(raw+filtered)
