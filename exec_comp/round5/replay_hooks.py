"""Full R5 hook replay on recorded states; only hardware/transport are stubbed."""
import argparse,json,time
from types import SimpleNamespace
import numpy as np
import torch
from exec_comp.common import ROOT,PROJECT,metadata,save_json
from exec_comp.round5.data import discover,load_run
from exec_comp.round5.adapter import EventAdapter
from exec_comp.round5.deployment import Hooks,LINKS
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from scalebridge.env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from scalebridge.online.adaptive_chunk_execution import StallLookahead
from scalebridge.online.current_pose import G1CurrentPoseFK
from scalebridge.online.reference import sample_chunk


class ReplayParent(HAT4AdaptiveMotionTrackingEnv):
    def _publish_robot_state(self):
        frame=self.fixture_state
        t=time.perf_counter_ns();link=self.current_pose_fk.compute(frame['p'],frame['quat'],frame['q'])
        self.fixture_fk_ms=(time.perf_counter_ns()-t)/1e6
        self.last_published_robot_state=dict(timestamp_ns=self.fixture_now,
            root_pos_world=frame['p'],root_quat_world_wxyz=frame['quat'],**link)
        self.simulator.dof_vel_tmp=frame['dq'];self.simulator.base_ang_vel_tmp=frame['w']
        self.state_buffer['root_pos_buffer'][0,-1]=torch.from_numpy(frame['p'])
        self.state_buffer['root_quat_wxyz_buffer'][0,-1]=torch.from_numpy(frame['quat'])
    def _apply_wrist_compensation(self,root):pass  # configurednone; omit old synchronous trace disk IO
    def _update_root_observation(self,root):pass  # measured replay root, no artificial transition


class Harness(Hooks,ReplayParent):pass


def initialize(method,run,folder):
    r=load_run(next(x for x in discover() if x['name']==run));env=Harness.__new__(Harness)
    env.device='cpu';env.dt=.02;env.cfg=dict(r['plan'].config);env.cfg['continuity_hand_event_guard']=False
    env.metadata_dict=metadata();env.active_indices={name:i for i,name in enumerate(metadata()['selected_body_names'])}
    env.current_pose_fk=G1CurrentPoseFK(PROJECT/'ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml',metadata()['joint_names'],link_names=metadata()['selected_body_names'])
    env.simulator=SimpleNamespace(dof_vel_tmp=np.zeros(29),base_ang_vel_tmp=np.zeros(3),sim_to_env_joint_idx=np.arange(29))
    env.future_offsets_s=np.arange(6)*.02;env.reference_plan_epoch=0
    env.focus_handoff=None;env.focus_adaptive=None;env.stall_wrist_assist=None;env.hand_event_playback_boundary=None
    env._focus_prediction_chunks={};env._adaptive_stall=StallLookahead();env._adaptive_key=env._adaptive_plan=None
    env._play_ns=env._play_real_ns=None;env._play_scale=1.;env.approach_slowdown=False;env.approach_slowdown_fingers_full_speed=True
    env.reference_play_gate=True;env.reference_playing=False;env.requires_restart=False;env.localization_paused=False;env.awaiting_fresh_chunk=False
    env.held_sample=env._latched_sample=env._latched_link_poses=None;env._prestart_link_poses=None;env.reference_forcing=False
    env.last_reference_sample=None;env.last_published_robot_state=None;env._r5_sequence=env._r5_epoch=None;env._r5_active=False
    env.state_buffer=dict(root_pos_buffer=torch.zeros(1,3,3),root_quat_wxyz_buffer=torch.zeros(1,3,4))
    env._r5=EventAdapter({'enable':True,'method':method},folder/'steps.jsonl')
    env._r5_ids=torch.tensor([env.active_indices[k] for k in LINKS]);return env,r


def run(method,duration,folder):
    folder.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    env,r=initialize(method,'20-31-41',folder)
    initial=r['plan'].starts[0];span=r['times'][-1]-initial-1
    previous_idx=None;starts=[];elapsed=[];stages=[];r1frame=50;clock=time.perf_counter();count=int(duration/.02)
    for k in range(count):
        deadline=clock+k*.02
        delay=deadline-time.perf_counter()
        if delay>0:time.sleep(delay)
        begin=time.perf_counter();starts.append(begin)
        recorded=initial+((k*.02)%max(span,1))
        index=int(np.searchsorted(r['times'],recorded).clip(0,len(r['times'])-1));event=int((np.searchsorted(r['plan'].stamps,recorded,side='right')-1).clip(0,len(r['plan'].chunks)-1))
        chunk=r['plan'].chunks[event];old=r['plan'].chunks[event-1] if event else None
        now_ns=time.monotonic_ns();env.fixture_now=now_ns
        env.fixture_state={key:r[arr][index] for key,arr in [('q','q'),('dq','dq'),('p','p'),('quat','quat'),('w','w')]}
        env.chunk=chunk['chunk'];env.chunk_started_ns=now_ns-int((chunk['start']/env.chunk['fps']+recorded-chunk['stamp'])*1e9)
        env.previous_chunk=None if old is None else old['chunk'];env.previous_chunk_started_ns=None if old is None else now_ns-int((old['start']/old['chunk']['fps']+recorded-old['stamp'])*1e9)
        env.blend_started_ns=now_ns-int((recorded-chunk['stamp'])*1e9);env.last_chunk_sequence_id=chunk['sequence']
        env.prestart_reference_sample=sample_chunk(env.chunk,chunk['start']/env.chunk['fps'],np.zeros(6))
        env.reference_playing=k>=r1frame
        env._publish_robot_state();publish_end=time.perf_counter()
        if k==r1frame:
            env.takeover_started_ns=now_ns;env.takeover_hold_dof_pos=torch.from_numpy(env.fixture_state['q'])
        if k>=r1frame:
            HAT4OnlineMotionTrackingEnv._select_applied_target(env,torch.from_numpy(env.fixture_state['q']),now_ns)
        env._gather_reference_state(now_ns)
        elapsed.append((time.perf_counter()-begin)*1000)
        stages.append(dict(env._r5_last_stages,publish_fk_history_ms=(publish_end-begin)*1000,fk_ms=env.fixture_fk_ms))
    env._r5.close();intervals=np.diff(starts)*1000
    def stat(values):return dict(median_ms=float(np.median(values)),mean_ms=float(np.mean(values)),p99_ms=float(np.percentile(values,99)),max_ms=float(np.max(values)))
    result=dict(method=method,duration_s=duration,frames=count,control_period=stat(intervals),callback=stat(elapsed),
        r1_first_interval_ms=float(intervals[r1frame]),stages={key:stat([x[key] for x in stages]) for key in stages[0]},
        worker_times=env._r5.worker_timings,log_drops=env._r5.writer.dropped,rejected_results=env._r5.rejected_results,
        acceptance=dict(median_period=bool(np.median(intervals)<=21),p99_period=bool(np.percentile(intervals,99)<=25),r1_interval=bool(intervals[r1frame]<=25)),
        scope='Actual R5 publish/gather hooks, existing CPU MuJoCo FK, original source reference sampler/gather and R1 blend selector; recorded accepted chunks/states replayed. Hardware,DDS,real policy inference,old synchronous legacy trace are excluded. CPU target buffers; GPU transfer acceptance N/A.',
        gpu_transfer='N/A: GPU occupied; this replay uses CPU target buffers',real_robot_run=False)
    save_json(folder/'metrics.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=('raw','a1','b1','all'),default='all');p.add_argument('--duration',type=float,default=120);a=p.parse_args()
    for method in ('raw','a1','b1') if a.method=='all' else (a.method,):
        result=run(method,a.duration,ROOT/'runs/round5/replay'/method)
        print(method,result['acceptance'],result['control_period'],flush=True)


if __name__=='__main__':main()
