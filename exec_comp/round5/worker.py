"""Network and100-frame construction live in a separate CPU process."""
import multiprocessing as mp
import queue
import time
import numpy as np
from exec_comp.common import ROOT


def _worker(method,inbox,outbox,ready):
    import torch
    from exec_comp.round5.reference import sample_snapshot,canonical9
    from exec_comp.round5.history import resample
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=torch.jit.load(str(ROOT/'releases/round5'/f'{method}_coefficients_cpu.pt'),map_location='cpu').eval()
    from exec_comp.round4.export import fixture
    h,w=fixture();w=w.contiguous()
    with torch.inference_mode():
        for _ in range(50):model(h,w)
    ready.set();calls=0
    while True:
        request=inbox.get()
        if request is None:break
        try:
            t=time.perf_counter_ns();history,history_info=resample(request['history'],request['stamp_ns']);th=time.perf_counter_ns()
            if 'window' in request:window=np.ascontiguousarray(request['window'],dtype=np.float32)
            else:window=canonical9(sample_snapshot(request['reference'],request['stamp_ns'],np.arange(100)/30))
            tw=time.perf_counter_ns();state={k:torch.from_numpy(v) for k,v in history.items()};goal=torch.from_numpy(window)
            with torch.inference_mode():coefficients=model(state,goal).numpy().copy()
            end=time.perf_counter_ns();calls+=1
            response=dict(sequence=request['sequence'],epoch=request['epoch'],stamp_ns=request['stamp_ns'],
                coefficients=coefficients,finite=bool(np.isfinite(coefficients).all()),history_info=history_info,
                timing_ms=dict(history=(th-t)/1e6,cpu_window=(tw-th)/1e6,network=(end-tw)/1e6,
                    cpu_gpu_transfer=0.,worker_total=(end-t)/1e6),first_real_call=calls==1)
        except Exception as exc:response=dict(sequence=request['sequence'],epoch=request['epoch'],error=repr(exc))
        try:outbox.put_nowait(response)
        except queue.Full:
            try:outbox.get_nowait()
            except queue.Empty:pass
            try:outbox.put_nowait(response)
            except queue.Full:pass


class CoefficientWorker:
    def __init__(self,method):
        if method not in ('a1','b1'):raise ValueError(method)
        ctx=mp.get_context('spawn');self.inbox=ctx.Queue(maxsize=1);self.outbox=ctx.Queue(maxsize=2);ready=ctx.Event()
        self.process=ctx.Process(target=_worker,args=(method,self.inbox,self.outbox,ready),daemon=True);self.process.start()
        if not ready.wait(30):raise RuntimeError('Coefficient worker warmup failed')
    def submit(self,request):
        try:self.inbox.put_nowait(request);return True
        except queue.Full:return False
    def poll(self):
        result=None
        for _ in range(2):
            try:result=self.outbox.get_nowait()
            except queue.Empty:break
        return result
    def close(self):
        try:self.inbox.put(None,timeout=1)
        except queue.Full:self.process.terminate()
        self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(1)
