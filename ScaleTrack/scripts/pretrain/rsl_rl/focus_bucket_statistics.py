"""Recover exact focus/non-focus per-motion sums from the existing evaluator."""
import math
import torch

METRICS = ['error_anchor_pos','error_anchor_rot'] + [
    f'error_{part}_{frame}_{kind}'
    for part in ('left_wrist','right_wrist','head_proxy')
    for frame in ('global','pelvis_local')
    for kind in ('pos','rot')
]

def partition(per_motion, lengths):
    lengths = lengths.float()
    # Existing evaluator accumulates elapsed < length, then divides the full
    # sum by length. Preserve that convention when recovering its raw sums.
    total_count = (lengths - 1).clamp_min(0)
    focus_count = per_motion['focus_frame_count'].float()
    other_count = total_count - focus_count
    if (other_count < 0).any():
        raise ValueError('Focus count exceeds evaluated frame count')
    result = dict(focus_count=focus_count, nonfocus_count=other_count)
    for key in METRICS:
        focus_sum = per_motion[key+'_focus'] * focus_count
        other_sum = per_motion[key] * lengths - focus_sum
        if (other_sum < -1.e-3).any():
            raise ValueError(f'Negative reconstructed non-focus sum: {key}')
        result[key+'_focus'] = per_motion[key+'_focus'].clone()
        result[key+'_nonfocus'] = other_sum.clamp_min(0) / other_count.clamp_min(1)
    return result

def summarize(partitioned, mask):
    output = {}
    for bucket in ('focus','nonfocus'):
        counts = partitioned[bucket+'_count']
        valid = mask & (counts > 0)
        values = {}
        for key in METRICS:
            scale = 180./math.pi if '_rot' in key else 100.
            error = partitioned[key+'_'+bucket][valid] * scale
            weight = counts[valid]
            values[key] = dict(mean_over_motions=float(error.mean()) if len(error) else None,
                              frame_weighted_mean=float((error*weight).sum()/weight.sum()) if len(error) else None)
        output[bucket] = dict(motions=int(valid.sum()),frames=int(counts[valid].sum()),errors_cm_deg=values)
    return output
