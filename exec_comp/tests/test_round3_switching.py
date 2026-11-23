"""The harness must behave like the deployed switcher, not like a copy of it."""
import json
import unittest
import numpy as np
from exec_comp.common import ROOT
from exec_comp.round3 import switching
from exec_comp.round3.switching import DeployedSwitching,make_chunk,open_fingertips

CONTRACT=json.loads((ROOT/"configs/online_execution_contract.json").read_text())
CONFIG={k:v["value"] for k,v in CONTRACT["switching_parameters"].items()}
FPS=CONTRACT["chunk_fps"];FRAMES=CONTRACT["chunk_frame_count"]
OFFSETS=np.arange(6)/50.


def straight_line(start_s,frames=FRAMES,fps=FPS,speed=.3,base=(0.,0.,.8)):
    """A steadily advancing reference, so travel and valley rules have input."""
    times=start_s+np.arange(frames)/fps
    positions=np.zeros((frames,4,3),dtype=np.float32)
    quaternions=np.zeros((frames,4,4),dtype=np.float32);quaternions[...,0]=1.
    positions[:,0]=np.stack([speed*times,np.zeros_like(times),np.full_like(times,base[2])],-1)
    positions[:,1]=positions[:,0]+np.array([0.,0.,.5],dtype=np.float32)
    positions[:,2]=positions[:,0]+np.array([0.,.2,.2],dtype=np.float32)
    positions[:,3]=positions[:,0]+np.array([0.,-.2,.2],dtype=np.float32)
    return positions,quaternions


def closing_hand(frames=FRAMES,close_at=40):
    """Fingertips that contract after ``close_at``, as the IK fallback reads."""
    tips=np.repeat(open_fingertips(frames)[:,None],2,axis=1)
    taper=np.clip((np.arange(frames)-close_at)/10.,0.,1.)[:,None,None,None]
    return (tips*(1.-taper)+tips*.2*taper).astype(np.float32)


class SwitchingHarnessTests(unittest.TestCase):
    def test_retained_packet_is_consumed_between_inference_arrivals(self):
        sw=self.switcher;second=None
        for step in range(220):
            now=step*20_000_000;p,q=straight_line(step*.02)
            seq=sw.publish_state(now,p[0],q[0])
            if step%20==0:
                result=sw.offer(make_chunk(step,seq,now,FPS,straight_line(step*.02)),now)
                if result['accepted'] and step>0:second=step;break
            result=sw.poll(now)
            if result['accepted'] and step>0:second=step;break
            if sw.env.chunk is not None:sw.sample(now)
        self.assertIsNotNone(second)
        self.assertNotEqual(second%20,0,'hold release must consume the retained chunk without waiting for next inference')

    def test_grasp_signal_exits_when_prediction_reopens(self):
        switcher=self.switcher
        positions,quats=straight_line(0.)
        switcher.publish_state(0,positions[0],quats[0])
        tips=np.repeat(open_fingertips(FRAMES)[:,None],2,axis=1)
        tips[15:45]*=.2
        switcher.offer(make_chunk(0,0,0,FPS,(positions,quats),tips),0)
        self.assertTrue(switcher.grasp_phase(int(25/FPS*1e9)))
        self.assertFalse(switcher.grasp_phase(int(60/FPS*1e9)))

    def setUp(self):
        self.switcher=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS)

    def feed(self,switcher,duration_s,replan_s=.408,source_age_s=.105,
             fingertips=None,measured=None,sequence=0):
        """Drive control steps at 50Hz, offering a chunk at the replan cadence."""
        accepted=[];samples=[];now=0;step=20_000_000;next_replan=0.
        for index in range(int(duration_s*50)):
            now=index*step
            time_s=index*.02
            positions,quaternions=straight_line(time_s)
            feedback=(positions[0],quaternions[0]) if measured is None else measured(time_s)
            switcher.publish_state(now,feedback[0],feedback[1])
            if time_s>=next_replan-1e-9:
                origin=max(0.,time_s-source_age_s)
                chunk=make_chunk(sequence,switcher.env.state_sequence_id-1,now,FPS,
                                 straight_line(origin),fingertips)
                # The source observation is source_age_s old.
                switcher.env.state_timestamp_history.append(
                    (switcher.env.state_sequence_id-1,int(now-source_age_s*1e9)))
                switcher.env.state_play_history.append(
                    (switcher.env.state_sequence_id-1,int(now-source_age_s*1e9)))
                result=switcher.offer(chunk,now)
                accepted.append(result);sequence+=1
                next_replan=time_s+replan_s
            if switcher.env.chunk is not None:samples.append(switcher.sample(now))
        return accepted,samples

    def test_vendor_is_the_deployed_code(self):
        self.assertTrue(switching.VENDOR["files"]>=18)
        env=self.switcher.env
        self.assertEqual(type(env).__name__,"HAT4AdaptiveMotionTrackingEnv")
        # The horizon and hold must come from the adaptive subclass, not the base.
        self.assertIs(type(env)._min_execution_hold_active,
                      type(env).__mro__[0].__dict__["_min_execution_hold_active"])
        self.assertEqual(env._chunk_execution_fraction(),0.)

    def test_first_chunk_enters_and_later_ones_are_held(self):
        accepted,_=self.feed(self.switcher,6.)
        self.assertTrue(accepted[0]["accepted"])
        held=[a for a in accepted[1:] if a["reason"]=="min_execution_hold"]
        self.assertTrue(held,"PACE must defer at least one replan")
        entered=[a for a in accepted if a["accepted"]]
        # 25-50 frames at 15Hz is 1.67-3.33s, so a 6s run cannot swap every
        # 0.41s replan, and must still swap more than once.
        self.assertGreater(len(entered),1)
        self.assertLess(len(entered),len(accepted))

    def test_switch_frames_match_the_pace_window(self):
        windows=[];reasons=set()
        switcher=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS)
        for index in range(int(20*50)):
            now=index*20_000_000;time_s=index*.02
            positions,quaternions=straight_line(time_s)
            switcher.publish_state(now,positions[0],quaternions[0])
            if switcher.env.chunk is None or index%21==0:
                chunk=make_chunk(switcher.env.last_chunk_sequence_id+1,
                                 switcher.env.state_sequence_id-1,now,FPS,
                                 straight_line(max(0.,time_s-.105)))
                switcher.env.state_timestamp_history.append(
                    (switcher.env.state_sequence_id-1,int(now-.105*1e9)))
                switcher.env.state_play_history.append(
                    (switcher.env.state_sequence_id-1,int(now-.105*1e9)))
                switcher.offer(chunk,now)
            plan=switcher.horizon(now)
            if plan is not None:
                windows.append(plan.frame-switcher.horizon_start_frame());reasons.add(plan.reason)
        self.assertTrue(windows)
        # Far cadence: between adaptive_far_min_frames and adaptive_far_max_frames
        # past the cursor the chunk was accepted at.
        self.assertGreaterEqual(min(windows),CONFIG["adaptive_far_min_frames"]-1e-6)
        self.assertLessEqual(max(windows),CONFIG["adaptive_far_max_frames"]+1e-6)
        self.assertTrue(reasons<={"speed_valley","bounded_fallback","motion_floor_unmet","event_boundary"})

    def test_close_boundary_caps_the_horizon(self):
        near=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS)
        self.feed(near,2.,fingertips=closing_hand(close_at=30))
        plan=near.horizon(int(1.98*50)*20_000_000)
        self.assertIsNotNone(plan)
        far=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS)
        self.feed(far,2.)
        far_plan=far.horizon(int(1.98*50)*20_000_000)
        self.assertIsNotNone(far_plan)
        # A chunk that closes at frame 30 cannot execute past that event.
        self.assertLessEqual(plan.frame,far_plan.frame)

    def test_open_fingertips_are_not_read_as_a_close(self):
        from scalebridge.online.adaptive_chunk_execution import closing_boundary
        chunk=make_chunk(0,0,0,FPS,straight_line(0.))
        self.assertEqual(closing_boundary(chunk,"right",0.,None),FRAMES-1)
        zeros=make_chunk(0,0,0,FPS,straight_line(0.),np.zeros((FRAMES,2,5,3),dtype=np.float32))
        self.assertEqual(closing_boundary(zeros,"right",0.,None),0)

    def test_overlap_blend_makes_the_seam_continuous(self):
        switcher=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS)
        before=after=None
        now=0;sequence=0;next_replan=0.;seams=[]
        for index in range(int(12*50)):
            now=index*20_000_000;time_s=index*.02
            positions,quaternions=straight_line(time_s)
            switcher.publish_state(now,positions[0],quaternions[0])
            if time_s>=next_replan-1e-9:
                before=None if switcher.env.chunk is None else switcher.sample(now)
                chunk=make_chunk(sequence,switcher.env.state_sequence_id-1,now,FPS,
                                 straight_line(max(0.,time_s-.105)))
                switcher.env.state_timestamp_history.append(
                    (switcher.env.state_sequence_id-1,int(now-.105*1e9)))
                switcher.env.state_play_history.append(
                    (switcher.env.state_sequence_id-1,int(now-.105*1e9)))
                if switcher.offer(chunk,now)["accepted"] and before is not None:
                    after=switcher.sample(now)
                    seams.append(float(np.linalg.norm(
                        after["right_wrist_pos"][0]-before["right_wrist_pos"][0])))
                sequence+=1;next_replan=time_s+.408
            if switcher.env.chunk is not None:switcher.sample(now)
        self.assertTrue(seams)
        # chunk_blend_s starts at alpha=0, so a swap is exactly continuous.
        self.assertLess(max(seams),1e-6)

    def test_anti_stall_lookahead_only_runs_when_enabled(self):
        def frozen(_time_s):
            positions,quaternions=straight_line(0.)
            return positions[0],quaternions[0]
        on=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS,anti_stall=True)
        self.feed(on,4.,measured=frozen)
        off=DeployedSwitching(config=CONFIG,future_offsets_s=OFFSETS,anti_stall=False)
        self.feed(off,4.,measured=frozen)
        self.assertGreater(max(on.env._adaptive_stall.budget.values()),0.)
        self.assertEqual(max(off.env._adaptive_stall.budget.values()),0.)
        self.assertLessEqual(max(on.env._adaptive_stall.budget.values()),
                             CONFIG["adaptive_stall_max_lookahead_m"]+1e-9)

    def test_reset_clears_every_clock(self):
        self.feed(self.switcher,3.)
        self.switcher.reset()
        self.assertIsNone(self.switcher.env.chunk)
        self.assertIsNone(self.switcher.env.chunk_started_ns)
        self.assertFalse(self.switcher.running())
        self.assertEqual(self.switcher.env.last_chunk_sequence_id,-1)


if __name__=="__main__":unittest.main()
