"""Absolute >5Hz wrist displacement RMS; saved traces only, no resimulation."""
import json
import numpy as np
from scipy.signal import welch
from exec_comp.common import ROOT,save_json


def displacement_rms(position,fs=50.):
    x=np.asarray(position,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,3) or len(x)<16 or not np.isfinite(x).all():
        raise ValueError('Need finite [time,2wrists,xyz], at least16frames')
    frequency,power=welch(x,fs=fs,axis=0,nperseg=min(256,len(x)),detrend='linear',window='hann',scaling='density')
    df=frequency[1]-frequency[0]
    # Integral of PSD has units m^2. Sum xyz, keep left/right separately.
    # Return frame-weighted energy to pool clips without concatenating them.
    variance=(power[frequency>5].sum(axis=(0,2))*df)
    return variance*len(x),len(x)



def collect():
    destination=ROOT/'runs/round4/brain_absolute_rms.json'
    # The local mirror carries JSON but intentionally omits large NPZs.
    # Keep valid remote-computed values when local traces are unavailable.
    output=json.loads(destination.read_text()).get('results',{}) if destination.exists() else {}
    for generation in ('round3','round4'):
        for path in sorted((ROOT/'runs'/generation/'brain_evaluation').glob('*/metrics.json')):
            row=json.loads(path.read_text())
            if row.get('truncated_smoke'):continue
            parts=[];missing=[]
            for index,episode in enumerate(row.get('per_clip',[])):
                if episode['type']!='robot':continue
                trace=path.parent/f'episode_{index:02d}.npz'
                if not trace.exists():missing.append(str(trace));continue
                with np.load(trace) as data:
                    try:parts.append(displacement_rms(data['actual'][:,2:4,:3]))
                    except ValueError:missing.append(str(trace))
            squared=np.sum([p[0] for p in parts],axis=0) if parts else None
            frames=sum(p[1] for p in parts)
            key=generation+'/'+path.parent.name
            if missing and output.get(key,{}).get('wrist_vector_rms_mm') is not None:continue
            output[key]=dict(
                wrist_vector_rms_mm=float(np.sqrt(squared.sum()/(2*frames))*1000) if frames and not missing else None,
                left_rms_mm=float(np.sqrt(squared[0]/frames)*1000) if frames and not missing else None,
                right_rms_mm=float(np.sqrt(squared[1]/frames)*1000) if frames and not missing else None,
                robot_clips=len(parts),frames=frames,missing=missing)
    result=dict(method='50Hz world-coordinate actual wrists; Welch PSD density:50Hz,Hann,linear detrend,nperseg=min(256,clip length),strictly>5Hz. Integrate PSD using frequency-bin width, sum xyz, average wrists, frame-weighted across independent clips, square root, convert m to mm. Same spectral estimator as existing energy-fraction gate; absolute RMS does not divide by total motion energy. Vector RMS, not per-axis RMS; gate unchanged.',results=output)
    save_json(ROOT/'runs/round4/brain_absolute_rms.json',result)
    return result


if __name__=='__main__':collect()
