import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from exec_comp.common import ROOT,metadata
from exec_comp.diff.fk import FK
from exec_comp.geometry import from_anchor
from exec_comp.models.body_wm import BodyWorldModel
from exec_comp.round3.transitions import TransitionWriter
from exec_comp.round3.audit_b1_model import score


class ReplayAuditTests(unittest.TestCase):
    def test_writer_and_paired_model_audit_use_same_next_K_not_library_target(self):
        torch.set_num_threads(2)
        temporary=ROOT/'runs/test_tmp';temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as path:
            origin=torch.tensor([[4.,0.,0.],[9.,0.,0.]])
            root=origin+torch.tensor([0.,0.,.8]);quat=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
            q=torch.tensor(metadata()['default_dof_pos'])[None].repeat(2,1)
            state64=torch.cat((q,torch.zeros(2,32),torch.tensor([[0.,0.,-1.]]).repeat(2,1)),-1)
            kp=from_anchor(FK().keypoints(q),root,quat)
            state=dict(q=q,dq=torch.zeros_like(q),w=torch.zeros(2,3),s=state64,kp=kp,
                root_p=root,root_q=quat,privileged=torch.tensor([[.8,0.,0.,0.]]).repeat(2,1))
            replay=SimpleNamespace(hist={'action':torch.zeros(2,3,29)},command=SimpleNamespace(
                motion_ids=torch.zeros(2,dtype=torch.long),time_steps=torch.zeros(2,dtype=torch.long)))
            target=kp.clone();target[:,3,0]+=.02
            writer=TransitionWriter(Path(path),'validation',2,origin,{})
            for _ in range(64):
                writer.before(state,replay,kp[:,None].repeat(1,6,1,1),kp,torch.zeros(2,29),
                    torch.zeros(2,29),torch.zeros(2,29),np.arange(2),np.zeros(2,dtype=bool),raw_target_next=target)
                writer.after(state,kp,torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
            writer.close()
            body=BodyWorldModel(True,16)
            with torch.no_grad():body.net[-1].weight.zero_();body.net[-1].bias.zero_()
            result=score(body,Path(path),'cpu',1)
            self.assertAlmostEqual(result['actual_wrist_cm'],1.,places=4)
            self.assertAlmostEqual(result['imagined_wrist_cm'],1.,places=4)
            self.assertLess(result['world_wrist_error_cm'],1e-4)  # float32 world-grid subtraction, <1 micron
            with np.load(writer.shards[0]) as shard:
                np.testing.assert_allclose(shard['root_pos'][...,0],0,atol=0)
            torch.testing.assert_close(state['root_p'],root,atol=0,rtol=0)
