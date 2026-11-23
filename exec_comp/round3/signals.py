"""Read-only grasp signal shared by simulation and the deployment wrapper."""
import numpy as np


def predicted_grasp_phase(chunk,cursor,hands,closing_boundary,first_boundaries=None):
    last=int(chunk['frame_count'])-1;frame=int(np.clip(cursor,0,last))
    for side in ('left','right'):
        hand=hands.get(side)
        if hand is not None:
            if any(e['kind']=='close' and e['start']<=cursor<=e['end'] for e in hand.get('events',())):return True
            continue
        if first_boundaries and first_boundaries[side]>=last and frame<last:continue
        boundary=closing_boundary(chunk,side,cursor,None)
        tips=np.asarray(chunk[f'{side}_fingertip_local'])[frame]
        closed=bool(np.all(np.linalg.norm(tips[1:]-tips[:1],axis=-1)<=.11))
        if (boundary<=cursor and boundary<last) or closed:return True
    return False
