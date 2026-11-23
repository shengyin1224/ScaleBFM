"""Separate learned dynamics drift from root-pose integration error."""
import torch
from exec_comp.common import ROOT, log, save_json
from exec_comp.datasets import Rollouts
from exec_comp.train_common import load_model
from exec_comp.diff.fk import FK
from exec_comp.geometry import from_anchor, integrate_root


def main():
    torch.set_num_threads(4)
    data = Rollouts("validation", body=True, device="cuda:0")
    model = load_model(ROOT/"runs/models/body_privileged_best.pt", "cuda:0").requires_grad_(False)
    fk = FK().cuda()
    batch = data.body_batch(1024, 167, privileged=True)
    result = {}
    with torch.no_grad():
        predicted = model.rollout(batch["s"][:, 0], batch["action"])
        root_p, root_q = batch["root_pos"], batch["root_quat"]
        old = batch["s"][:, 0]
        for step in range(167):
            truth = batch["next_s"][:, step]
            root_p, root_q = integrate_root(root_p, root_q, old, truth)
            if step+1 in (1, 8, 32, 167):
                state = predicted[:, step]
                local = fk.keypoints(state[:, :29])
                # Oracle root isolates joint-model errors; oracle state tests integration itself.
                oracle_root = from_anchor(local, batch["next_root_pos"][:, step], batch["next_root_quat"][:, step])
                integrated_truth = from_anchor(fk.keypoints(truth[:, :29]), root_p, root_q)
                target = batch["ach"][:, step]
                result[str(step+1)] = {
                    "learned_joint_wrist_error_with_oracle_root_cm": ((oracle_root[:, 2:, :3]-target[:, 2:, :3]).norm(dim=-1).mean(0)*100).tolist(),
                    "oracle_state_integration_wrist_error_cm": ((integrated_truth[:, 2:, :3]-target[:, 2:, :3]).norm(dim=-1).mean(0)*100).tolist(),
                    "predicted_gravity_norm_mean": float(state[:, 61:64].norm(dim=-1).mean()),
                    "truth_gravity_norm_mean": float(truth[:, 61:64].norm(dim=-1).mean()),
                    "state_mae": {name: float((state[:, a:b]-truth[:, a:b]).abs().mean()) for name, a, b in (("q_rad",0,29),("dq_rad_s",29,58),("ang_rad_s",58,61),("gravity",61,64),("height_m",64,65),("velocity_m_s",65,68))},
                }
            old = truth
    save_json(ROOT/"configs/body_rollout_diagnosis.json", result)
    log("body_rollout_diagnosis", horizons=result)


if __name__ == "__main__":
    main()
