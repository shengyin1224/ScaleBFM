"""Control adapter: cache measured CPU state, submit event, expand available curve."""
import time
import numpy as np
from exec_comp.round5.engine import CurveExecutor,Curve,rotate,qinv
from exec_comp.round5.history import TimestampHistory
from exec_comp.round5.worker import CoefficientWorker
from exec_comp.round5.async_io import AsyncWriter


class EventAdapter:
    def __init__(self,options,log_path=None):
        self.enabled=bool(options.get('enable',False));self.method=str(options.get('method','raw'))
        self.history=TimestampHistory();self.engine=CurveExecutor(enable=self.enabled and self.method!='raw',
            cap_m=float(options.get('total_offset_cap_m',.05)),max_step_m=float(options.get('max_step_cm',1.))/100,
            angular_step_deg=float(options.get('angular_step_deg',3.)),angular_rate_deg_s=float(options.get('angular_rate_deg_s',150.)))
        self.worker=None;self.writer=None;self.sequence=None;self.epoch=0;self.active=False;self.pending=None
        self.worker_timings=[];self.control_timings=[];self.rejected_results=0
        if not self.enabled:return
        if self.method not in ('raw','a1','b1'):raise ValueError('method must be raw|a1|b1')
        if self.method!='raw':self.worker=CoefficientWorker(self.method)  #50 warmups before robot init/R1
        if log_path is not None:self.writer=AsyncWriter(log_path)

    def observe(self,stamp_ns,state):
        if self.enabled:self.history.append(stamp_ns,state)

    def event(self,stamp_ns,sequence,epoch,*,reference=None,window=None):
        if not self.enabled or self.method=='raw':return
        self.sequence=sequence;self.epoch=epoch
        request=dict(stamp_ns=int(stamp_ns),sequence=int(sequence),epoch=int(epoch),history=self.history.snapshot())
        if window is not None:request['window']=np.ascontiguousarray(window,dtype=np.float32)
        else:request['reference']=reference
        self.pending=request

    def correct(self,stamp_ns,deployed,raw,*,sequence,epoch,grasp=False,active=True,offsets=None):
        if not self.enabled:return deployed,{}
        start=time.perf_counter_ns()
        bad_result=False
        if self.worker is not None:
            if self.pending is not None and self.worker.submit(self.pending):self.pending=None
            response=self.worker.poll()
            if response is not None:
                if response.get('sequence')!=sequence or response.get('epoch')!=epoch:self.rejected_results+=1
                elif 'error' in response or not response.get('finite',False):
                    bad_result=True
                else:
                    curve=Curve(response['coefficients'],response['stamp_ns'],sequence,epoch)
                    self.engine.install(curve,stamp_ns);self.worker_timings.append(response['timing_ms'])
        wrong_epoch=self.engine.curve is not None and self.engine.curve.epoch!=epoch
        guard_start=time.perf_counter_ns()
        output,telemetry=self.engine.step(stamp_ns,deployed,raw,grasp=grasp,offsets=offsets,force_fallback=not active or bad_result or wrong_epoch)
        guard_end=time.perf_counter_ns()
        self.active=active
        elapsed=(time.perf_counter_ns()-start)/1e6;self.control_timings.append(elapsed)
        telemetry.update(method=self.method,sequence=sequence,epoch=epoch,control_compensation_ms=elapsed,bad_worker_result=bad_result,
            protection_expand_ms=(guard_end-guard_start)/1e6,ipc_poll_ms=(guard_start-start)/1e6)
        return output,telemetry

    def log(self,stamp_ns,raw,corrected,actual,telemetry):
        if self.writer is None:return
        self.writer.submit(dict(stamp_ns=int(stamp_ns),wall_time_ns=time.time_ns(),method=self.method,
            raw_position=raw[0,:,:3].copy(),raw_quaternion=raw[0,:,3:7].copy(),
            corrected_position=corrected[0,:,:3].copy(),corrected_quaternion=corrected[0,:,3:7].copy(),
            actual_position=actual[...,:3].copy(),actual_quaternion=actual[...,3:7].copy(),telemetry=telemetry))

    def close(self):
        if self.worker:self.worker.close()
        if self.writer:self.writer.close()
