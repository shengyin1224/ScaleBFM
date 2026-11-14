"""Checks that protect the deployment contract and teacher/student masking."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from policy import mean_action, disable_phase_weights


class EchoPolicy:
    actor_task_embedder = staticmethod(lambda x: x)
    actor = staticmethod(lambda prop, action, tokens: tokens.sum((1, 2))[:, None])


class ContractTests(unittest.TestCase):
    def observations(self):
        mode = torch.zeros(1, 14)
        mode[:, [0,7,10,13]] = 1
        mapping = torch.cat([mode.repeat_interleave(n, dim=1) for n in (3,3,6,6)]
                            + [torch.ones(1,3)], dim=1)
        task = torch.zeros(1,6,255)
        task[:,:,1*3] = 2  # inactive link must not reach HAT-4 actor
        return {"policy": torch.zeros(1,3,64), "teacher_policy": torch.zeros(1,3,64),
                "policy_task": task, "teacher_policy_task": task.clone(),
                "mode": mode, "mode_mapping": mapping, "action": torch.zeros(1,3,29)}

    def test_teacher_sees_inactive_links_student_does_not(self):
        obs = self.observations()
        student = mean_action(EchoPolicy(), obs)
        teacher = mean_action(EchoPolicy(), obs, teacher=True)
        self.assertEqual(float(student), 6*4)
        self.assertEqual(float(teacher), 6*(14+2))
        self.assertEqual(float(obs["policy_task"][0,0,3]), 2)

    def test_phase_cannot_change_student_action(self):
        obs = self.observations()
        before = mean_action(EchoPolicy(), obs)
        obs["policy_task"][:,:,253:255] = 123
        torch.testing.assert_close(before, mean_action(EchoPolicy(), obs))

    def test_zero_phase_columns_preserve_mode_columns(self):
        p = SimpleNamespace(actor_task_embedder=SimpleNamespace(task_projection=torch.nn.Linear(269,8)),
                            critic_task_embedder=SimpleNamespace(task_projection=torch.nn.Linear(339,8)))
        mode_weights = p.actor_task_embedder.task_projection.weight[:,255:].clone()
        disable_phase_weights(p)
        self.assertEqual(float(p.actor_task_embedder.task_projection.weight[:,253:255].abs().sum()), 0)
        torch.testing.assert_close(mode_weights, p.actor_task_embedder.task_projection.weight[:,255:])

    def test_combined_cap_and_curriculum(self):
        source = Path(__file__).with_name("env_cfg.py").read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="wrist_precision")
        helper = Path(__file__).resolve().parents[2] / "source/scaletrack/scaletrack/tasks/tracking/mdp/reference_playback.py"
        spec = importlib.util.spec_from_file_location("playback_helpers", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ns = {"eased_progress": module.eased_progress,
              "adaptive_wrist_precision": lambda *a,**kw: torch.tensor(kw["w_pos"]+kw["w_rot"])}
        exec(compile(ast.Module(body=[fn],type_ignores=[]), "reward", "exec"), ns)
        cmd = SimpleNamespace(training_steps=0, cfg=SimpleNamespace(precision_start_step=128000,precision_end_step=512000))
        env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd))
        for cap in (1.,2.):
            for step, expected in ((0,0),(128000,0),(320000,.75*cap),(512000,cap),(896000,cap)):
                cmd.training_steps=step
                self.assertAlmostEqual(float(ns["wrist_precision"](env,cap=cap)),expected)


if __name__ == "__main__":
    unittest.main()
