"""Memoize the ORIGINAL pure closing_boundary on its complete effective inputs.

Simulation only. The original function uses start solely through clipped
floor(start), frame_count, fingertip values and the hand event kind/start/end.
Cache keys include every fingertip byte (not a lossy digest) and the event
values. Mutation of a chunk or event changes its key. A miss calls the original
function itself; no boundary selection is reimplemented here. Never installed
by the real deployment adapter.
"""
from collections import OrderedDict
import numpy as np


class BoundaryCache:
    def __init__(self,original,capacity=8192):
        self.original=original;self.capacity=capacity;self.values=OrderedDict();self.hits=0;self.misses=0

    def __call__(self,chunk,side,start=0.,hand=None):
        n=int(chunk['frame_count']);begin=int(np.clip(np.floor(start),0,n-1))
        tips=chunk.get(f'{side}_fingertip_local')
        data=np.asarray(tips,dtype=float) if tips is not None else None
        events=None if hand is None else tuple(
            (repr(e.get('kind')),repr(e.get('start')),repr(e.get('end'))) for e in hand.get('events',()))
        key=(n,side,begin,None if data is None else (data.shape,data.tobytes()),events)
        if key in self.values:
            self.hits+=1;self.values.move_to_end(key);return self.values[key]
        self.misses+=1
        answer=self.original(chunk,side,start,hand)
        self.values[key]=answer
        if len(self.values)>self.capacity:self.values.popitem(last=False)
        return answer

    def report(self):
        return dict(hits=self.hits,misses=self.misses,capacity=self.capacity,
            policy='original function results, exact full fingertip bytes+events+clipped floor cursor; simulation only')


def install():
    from pathlib import Path
    from exec_comp.round3.vendor import VENDOR
    from scalebridge.online import adaptive_chunk_execution as core
    from scalebridge.env import motion_tracking_hat4_adaptive as env
    if not Path(env.__file__).is_relative_to(VENDOR):raise RuntimeError('Memoization is allowed only on the simulation snapshot')
    if isinstance(core.closing_boundary,BoundaryCache):return core.closing_boundary
    cache=BoundaryCache(core.closing_boundary)
    core.closing_boundary=cache;env.closing_boundary=cache
    return cache
