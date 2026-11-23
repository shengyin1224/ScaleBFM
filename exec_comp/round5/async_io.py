"""Isolated writer process; control does not serialize JSON or wait for disk."""
import multiprocessing as mp
import queue
import json
from pathlib import Path
import numpy as np
import time


def _default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def _write(path,inbox,ready):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',buffering=65536) as stream:
        durations=[]
        ready.set()
        while True:
            item=inbox.get()
            if item is None:break
            start=time.perf_counter_ns()
            stream.write(json.dumps(item,default=_default,allow_nan=False,separators=(',',':'))+'\n')
            durations.append((time.perf_counter_ns()-start)/1e6)
    if durations:
        path.with_suffix('.profile.json').write_text(json.dumps(dict(rows=len(durations),mean_ms=float(np.mean(durations)),p99_ms=float(np.percentile(durations,99)),location='isolated writer process, never control thread')))


class AsyncWriter:
    def __init__(self,path):
        ctx=mp.get_context('spawn');self.inbox=ctx.Queue(maxsize=256);ready=ctx.Event();self.dropped=0
        self.process=ctx.Process(target=_write,args=(str(path),self.inbox,ready),daemon=True);self.process.start()
        if not ready.wait(15):raise RuntimeError('Log writer did not initialize')
    def submit(self,row):
        try:self.inbox.put_nowait(row);return True
        except queue.Full:self.dropped+=1;return False
    def close(self):
        try:self.inbox.put(None,timeout=2)
        except queue.Full:self.process.terminate()
        self.process.join(timeout=5)
        if self.process.is_alive():self.process.terminate();self.process.join(1)
