"""CPU contract checks for exact detector parity, hand separation and reward gating."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/root/shengyin/ScaleBFM_focus_brain_20261114')
RUN = ROOT / 'experiments/focus_brain_20261114'
builder_file = ROOT / 'ScaleTrack/scripts/pretrain/data_process/build_brain_focus_lr.py'
spec = importlib.util.spec_from_file_location('brain_label_builder', builder_file)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
detector, digest = builder.load_brain_detector('/mnt/nvme0n1/human_policy_ckpt/isolated_20261021_no_left_wrist/src/hdt/auxiliary_tasks.py')

def state_from_spreads(left, right):
    states = np.zeros((len(left),128),np.float32)
    for signal, offset in ((left,10),(right,40)):
        tips = states[:,offset:offset+18].reshape(-1,6,3)
        for finger in range(2,6):
            tips[:,finger,0] = signal
    return states

fps = 30.
left = np.r_[np.full(60,.08),np.linspace(.08,.02,30),np.full(60,.02),np.linspace(.02,.08,30),np.full(60,.08)]
right = np.full(len(left),.08)
_,_,weights,events = detector(state_from_spreads(left,right),fps)
assert len(events) == 2 and all(e[0] == 0 for e in events)
assert (weights[:,1] == 1).all() and (weights[:,0] > 1).any()
for hand, _, onset, confirmed in events:
    expected = np.arange(max(0,onset-int(.3*fps)),min(len(left),confirmed+int(.3*fps)+1))
    assert (weights[expected,hand] == 5).all()
_,_,swapped,_ = detector(state_from_spreads(right,left),fps)
assert np.array_equal(weights[:,::-1],swapped)
assert builder.next_edge(np.array([False,False,True,True,False]),1.).tolist() == [2.,1.,-2.,-1.,1.]

module = ROOT / 'ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/mdp/brain_focus_lr.py'
tree = ast.parse(module.read_text())
node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'weighted_errors_lr')
scope = {'torch':torch}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(module),'exec'),scope)
names = ['pelvis','left_wrist_yaw_link','right_wrist_yaw_link','torso_link']
errors = torch.tensor([[1.,2.,3.,4.]]).repeat(4,1)
flags = torch.tensor([[False,False],[True,False],[False,True],[True,True]])
actual = scope['weighted_errors_lr'](errors,names,flags,2.)
expected = torch.tensor([10.,16.,17.,19.])/4.
torch.testing.assert_close(actual,expected)
assert torch.equal(scope['weighted_errors_lr'](errors,names,flags,1.),errors.mean(-1))

# A dataset switch happens before evaluation resets the previous motion IDs.
# Capability detection must not call a property with those transient IDs.
runner_file = ROOT / 'ScaleTrack/source/my_rsl_rl/my_rsl_rl/runners/on_policy_runner.py'
runner_tree = ast.parse(runner_file.read_text())
check = next(n for n in ast.walk(runner_tree) if isinstance(n,ast.Assign)
             and isinstance(n.value,ast.Call)
             and any(isinstance(t,ast.Name) and t.id == 'independent_focus' for t in n.targets))
class SwitchingCommand:
    @property
    def focus_flag_lr(self):
        raise IndexError('old train motion id exceeds held-out set')
expression = compile(ast.Expression(body=check.value), str(runner_file), 'eval')
assert eval(expression, {'command':SwitchingCommand()}) is True
assert eval(expression, {'command':object()}) is False

# Exercise the actual actor inference implementation without starting a GPU.
actor_file = ROOT / 'ScaleTrack/source/my_rsl_rl/my_rsl_rl/modules/actor_critic_humanoid_transformer.py'
actor_tree = ast.parse(actor_file.read_text())
inference = next(n for n in ast.walk(actor_tree) if isinstance(n,ast.FunctionDef) and n.name == 'act_inference')
scope = {'torch':torch,'TensorDict':object}
exec(compile(ast.Module(body=[inference],type_ignores=[]),str(actor_file),'exec'),scope)
class DummyActor:
    state_dependent_std = False
    def get_actor_obs(self, obs, inference):
        self.inference = inference
        return None, torch.zeros(1,4), None
    def actor_task_embedder(self, task):
        return task
    def actor(self, prop, action, task):
        return torch.zeros(1,29)
dummy = DummyActor()
scope['act_inference'](dummy,{})
assert dummy.inference is True
scope['act_inference'](dummy,{},respect_mode=True)
assert dummy.inference is False

audit = json.loads((RUN / 'corpus/audit.json').read_text())
assert audit['episodes'] == 3000 and audit['detector_ast_sha256'] == digest
assert (RUN / 'corpus/READY').read_text().strip() == digest
report = dict(detector_parity=True, independent_hands=True, no_stationary_hand_events=True,
              exact_window_padding=True, signed_edges=True, independent_reward_gates=True,
              torso_union_gate=True, dataset_switch_property_guard=True, evaluation_respects_sparse_mode=True, validated_corpus_episodes=audit['episodes'])
(RUN / 'contract_test.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
