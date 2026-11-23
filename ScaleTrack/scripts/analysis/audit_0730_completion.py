#!/usr/bin/env python3
"""Machine-check the evidence required by ``0730-next-steps.md``.

Use ``--allow-pending-xl`` while the official XL-F3h run is still active.  The
final acceptance run intentionally has no such flag and exits non-zero unless
the XL model_20000 evaluation and archive are present and internally
consistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


TRACK = Path(__file__).resolve().parents[2]
ROOT = TRACK.parent
RESULTS = TRACK / "action_error_results"
VARIANTS = ("F1", "F2", "F2o", "F3", "F3h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-xl", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "0730_completion_audit.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def motion_failures(diagnostics: dict) -> set[str]:
    return {
        entry["motion"]
        for entry in diagnostics["motions_detail"]
        if entry["failed"]
    }


def add(checks: list[dict], name: str, passed: bool, evidence: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    args = parse_args()
    checks: list[dict] = []

    # Section 1: structural failure audit and causal crop retest.
    autopsy = RESULTS / "failed_115_autopsy"
    overlap = load_json(autopsy / "failure_overlap_and_reference_audit.json")
    failure_list_paths = sorted(
        (RESULTS / "heldout_600_fixed_mode").glob(
            "*_runtime/failed_train_motions.txt"
        )
    )
    failure_sets = [
        set(filter(None, path.read_text().splitlines())) for path in failure_list_paths
    ]
    recomputed_common = set.intersection(*failure_sets)
    recomputed_union = set.union(*failure_sets)
    pairwise_jaccards = []
    for left_index, left in enumerate(failure_sets):
        for right in failure_sets[left_index + 1 :]:
            pairwise_jaccards.append(len(left & right) / len(left | right))
    category_common_total = sum(
        row["common_failed"] for row in overlap["category_distribution"].values()
    )
    reference_contingency = overlap["reference_quality_contingency"]
    add(
        checks,
        "failure_overlap",
        overlap["variant_count"] == 7
        and overlap["common_failure_count"] == 114
        and overlap["union_failure_count"] == 118
        and len(failure_list_paths) == 7
        and len(recomputed_common) == 114
        and len(recomputed_union) == 118
        and recomputed_common == set(overlap["common_failures"])
        and category_common_total == 114
        and math.isclose(min(pairwise_jaccards), 114 / 118)
        and math.isclose(max(pairwise_jaccards), 1.0)
        and reference_contingency["common_failures"] == {
            "suspicious": 111,
            "total": 114,
        }
        and reference_contingency["all_motions"] == {
            "suspicious": 147,
            "total": 600,
        }
        and set(overlap["representative_motions"]).issubset(recomputed_common),
        {
            "variants": overlap["variant_count"],
            "common": overlap["common_failure_count"],
            "union": overlap["union_failure_count"],
            "recomputed_common": len(recomputed_common),
            "recomputed_union": len(recomputed_union),
            "pairwise_jaccard_min_max": [
                min(pairwise_jaccards),
                max(pairwise_jaccards),
            ],
            "category_common_total": category_common_total,
            "suspicious_common": reference_contingency["common_failures"],
            "suspicious_all": reference_contingency["all_motions"],
        },
    )
    original = load_json(autopsy / "released_hat4_original_diagnostics.json")
    repaired = load_json(autopsy / "released_hat4_repaired_v2_diagnostics.json")
    original_failed = motion_failures(original)
    repaired_failed = motion_failures(repaired)
    add(
        checks,
        "failure_timing_and_links",
        original["failed_motion_count"] == 115
        and original["failure_timing"]["early_le_1s"] == 115
        and sum(original["dominant_group_at_first_failure"].values()) == 115,
        {
            "failed": original["failed_motion_count"],
            "early": original["failure_timing"]["early_le_1s"],
            "groups": original["dominant_group_at_first_failure"],
        },
    )
    repair_report = load_json(
        TRACK
        / "data/hat4_unifolm/manifests/unifolm_hat4_heldout_repaired_v2.repair.json"
    )
    repair_source_manifest = yaml.safe_load(Path(repair_report["source_manifest"]).read_text())
    repair_output_manifest = yaml.safe_load(Path(repair_report["output_manifest"]).read_text())
    repair_records = repair_report["records"]
    repaired_records = [row for row in repair_records if row["status"] == "repaired"]
    unchanged_records = [
        row for row in repair_records if row["status"] == "selected_but_no_discontinuity"
    ]
    repaired_payloads_valid = True
    for row in repaired_records:
        source_path = Path(row["source"])
        target_path = Path(row["target"])
        start = row["removed_prefix_frames"]
        if (
            source_path.resolve() == target_path.resolve()
            or not source_path.is_file()
            or not target_path.is_file()
            or Path(repair_output_manifest[row["motion"]]).resolve()
            != target_path.resolve()
            or not 1 <= start <= repair_report["thresholds"]["max_prefix_frames"]
            or row["original_frames"] - start != row["repaired_frames"]
        ):
            repaired_payloads_valid = False
            continue
        with np.load(source_path, allow_pickle=False) as source_motion, np.load(
            target_path, allow_pickle=False
        ) as target_motion:
            body_position = np.asarray(target_motion["body_pos_w"], dtype=np.float64)
            body_step = np.linalg.norm(np.diff(body_position, axis=0), axis=-1)
            repaired_payloads_valid &= (
                target_motion["joint_pos"].shape[0] == row["repaired_frames"]
                and np.array_equal(target_motion["joint_pos"], source_motion["joint_pos"][start:])
                and np.array_equal(target_motion["body_pos_w"], source_motion["body_pos_w"][start:])
                and not (
                    (body_step[:, 0] > repair_report["thresholds"]["root_step_m"])
                    | (
                        body_step.max(axis=1)
                        > repair_report["thresholds"]["body_step_m"]
                    )
                ).any()
            )
    unchanged_payloads_valid = all(
        Path(repair_output_manifest[row["motion"]]).resolve()
        == Path(row["source"]).resolve()
        for row in unchanged_records
    )
    selected_record_names = {row["motion"] for row in repair_records}
    unselected_payloads_valid = all(
        Path(repair_output_manifest[name]).resolve() == Path(source_path).resolve()
        for name, source_path in repair_source_manifest.items()
        if name not in selected_record_names
    )
    add(
        checks,
        "crop_counterfactual",
        repair_report["repaired_motions"] == 102
        and repair_report["unchanged_selected_motions"] == 12
        and len(repair_source_manifest) == 600
        and len(repair_output_manifest) == 600
        and len(repaired_records) == 102
        and len(unchanged_records) == 12
        and repaired_payloads_valid
        and unchanged_payloads_valid
        and unselected_payloads_valid
        and repaired["failed_motion_count"] == 115
        and original_failed == repaired_failed,
        {
            "repaired_motions": repair_report["repaired_motions"],
            "unchanged_selected_motions": repair_report[
                "unchanged_selected_motions"
            ],
            "independent_payloads_and_manifest_valid": repaired_payloads_valid
            and unchanged_payloads_valid
            and unselected_payloads_valid,
            "original_failed": len(original_failed),
            "repaired_failed": len(repaired_failed),
            "same_failure_set": original_failed == repaired_failed,
        },
    )
    videos = list((autopsy / "representative_renders/videos").glob("*/*.mp4"))
    sheets = list((autopsy / "representative_renders/contact_sheets").glob("*.png"))
    render_summary = load_json(autopsy / "representative_renders/render_summary.json")
    add(
        checks,
        "representative_renders",
        len(videos) == 10
        and len(sheets) == 10
        and all(path.stat().st_size > 100_000 for path in videos)
        and all(path.stat().st_size > 10_000 for path in sheets)
        and len(render_summary["results"]) == 10
        and all(row["exit_code"] == 0 for row in render_summary["results"]),
        {
            "videos": len(videos),
            "contact_sheets": len(sheets),
            "minimum_video_bytes": min(
                (path.stat().st_size for path in videos), default=0
            ),
            "minimum_contact_sheet_bytes": min(
                (path.stat().st_size for path in sheets), default=0
            ),
        },
    )

    # Section 2: all M runs, rolling fixed-HAT4 evaluation, winner and release.
    formal_root = TRACK / "logs/rsl_rl/g1_hat4_focus_finetune"
    final_checkpoints = {
        variant: formal_root
        / f"formal_core5_20260728b_{variant}/model_20000.pt"
        for variant in VARIANTS
    }
    add(
        checks,
        "m_runs_reached_20k",
        all(path.is_file() and path.stat().st_size > 0 for path in final_checkpoints.values()),
        {name: str(path) for name, path in final_checkpoints.items()},
    )
    rolling = list((RESULTS / "heldout_600_fixed_mode_checkpoints").glob("*/model_*.json"))
    rolling_expected = {
        (variant, step)
        for variant in VARIANTS
        for step in (12000, 16000, 20000)
    }
    rolling_actual = {
        (path.parent.name, int(path.stem.removeprefix("model_"))) for path in rolling
    }
    summary = load_json(RESULTS / "final_checkpoint_selection/fixed_checkpoint_summary.json")
    summary_by_variant_step = {
        (row["variant"], row["step"]): row for row in summary["rows"]
    }
    rolling_reports_valid = True
    rolling_report_evidence = []
    expected_heldout_manifest = (
        TRACK / "data/hat4_unifolm/manifests/unifolm_hat4_heldout.yaml"
    ).resolve()
    for path in rolling:
        variant = path.parent.name
        step = int(path.stem.removeprefix("model_"))
        report = load_json(path)
        checkpoint_path = Path(report["checkpoint"])
        failed_list_path = Path(report["failed_motions_file"])
        failed_list_count = (
            len(list(filter(None, failed_list_path.read_text().splitlines())))
            if failed_list_path.is_file()
            else -1
        )
        row = summary_by_variant_step.get((variant, step))
        metric_values = [
            float(report["metrics"][key])
            for key in (
                "error_left_wrist_pelvis_local_pos_focus_success",
                "error_left_wrist_pelvis_local_rot_focus_success",
                "error_right_wrist_pelvis_local_pos_focus_success",
                "error_right_wrist_pelvis_local_rot_focus_success",
                "error_head_proxy_pelvis_local_pos_focus_success",
                "error_head_proxy_pelvis_local_rot_focus_success",
            )
        ]
        valid = (
            report["simulator"] == "IsaacLab"
            and checkpoint_path.is_file()
            and checkpoint_path.name == f"model_{step}.pt"
            and checkpoint_path.parent.name.endswith(f"_{variant}")
            and Path(report["motion_file"]).resolve() == expected_heldout_manifest
            and report["motions"] == 600
            and report["eval_max_steps_per_motion"] == 1000
            and report["mode"]["index"] == 8
            and report["mode"]["name"] == "HAT-4"
            and failed_list_count == report["failed_motion_count"]
            and row is not None
            and row["passed"] == 600 - report["failed_motion_count"]
            and all(math.isfinite(value) and value >= 0.0 for value in metric_values)
        )
        rolling_reports_valid &= valid
        rolling_report_evidence.append(
            {
                "variant": variant,
                "step": step,
                "passed": 600 - report["failed_motion_count"],
                "valid": valid,
            }
        )
    rows_20k = {row["variant"]: row for row in summary["rows"] if row["step"] == 20000}
    add(
        checks,
        "rolling_fixed_hat4",
        rolling_expected == rolling_actual
        and rolling_reports_valid
        and len(rows_20k) == 5
        and all(row["total"] == 600 for row in rows_20k.values()),
        {
            "expected_jsons": len(rolling_expected),
            "actual_jsons": len(rolling_actual),
            "reports": rolling_report_evidence,
            "passes_20k": {name: row["passed"] for name, row in rows_20k.items()},
        },
    )
    release_m = TRACK / "releases/hat4_focus_release_v1/model_20000.pt"
    release_m_sha = sha256(release_m)
    release_m_recipe = load_json(TRACK / "releases/hat4_focus_release_v1/recipe.json")
    f3_20k = rows_20k["F3"]
    f3h_20k = rows_20k["F3h"]
    wrist_rotation_disadvantages_deg = {
        side: f3h_20k[f"error_{side}_wrist_pelvis_local_rot_focus_success"]
        - f3_20k[f"error_{side}_wrist_pelvis_local_rot_focus_success"]
        for side in ("left", "right")
    }
    worst_wrist_rotation_disadvantage_deg = max(
        wrist_rotation_disadvantages_deg.values()
    )
    add(
        checks,
        "m_winner_release",
        release_m_sha
        == "c9c65cfb60bea8e4a254abaa8a60496e8fd0b9703751a0d70c3d322789766393"
        and worst_wrist_rotation_disadvantage_deg < 1.0
        and release_m_recipe["variant"] == "F3h"
        and release_m_recipe["sha256"] == release_m_sha
        and release_m_recipe["heldout_fixed_hat4"]["passed"] == f3h_20k["passed"]
        and release_m_recipe["heldout_fixed_hat4"]["total"] == 600,
        {
            "winner": "F3h",
            "sha256": release_m_sha,
            "wrist_rotation_disadvantages_deg": wrist_rotation_disadvantages_deg,
            "worst_disadvantage_deg": worst_wrist_rotation_disadvantage_deg,
            "predefined_threshold_deg": 1.0,
            "f3_5_required": worst_wrist_rotation_disadvantage_deg >= 1.0,
        },
    )

    # Section 3: dynamic MuJoCo re-anchor evidence.
    reanchor = load_json(RESULTS / "reanchor_dynamic/reanchor_ablation_summary.json")
    reanchor_rows = {row["run"]: row for row in reanchor["rows"]}
    required_runs = {"static", "drift5_on", "drift5_off", "drift10_on", "drift10_off"}
    reanchor_reports = {
        name: load_json(RESULTS / f"reanchor_dynamic/{name}.json")
        for name in required_runs
    }
    expected_offsets = {
        "static": 0.0,
        "drift5_on": 0.05,
        "drift5_off": 0.05,
        "drift10_on": 0.10,
        "drift10_off": 0.10,
    }
    reanchor_semantics_valid = all(
        report["simulator"] == "MuJoCo"
        and report["control_dt"] == 0.02
        and report["reanchor_ablation"]["evaluation_reference"]
        == "original fixed world reference"
        and report["reanchor_ablation"]["injected_initial_root_drift_xyz_m"]
        == [expected_offsets[name], 0.0, 0.0]
        and report["reanchor_ablation"]["policy_reference"]
        == (
            "reference translated by injected drift (stale root-local target)"
            if name.endswith("_off")
            else "original fixed world reference"
        )
        and report["evaluated_transitions"] == (250 if name == "static" else 774)
        for name, report in reanchor_reports.items()
    )
    off_10 = max(
        reanchor_rows["drift10_off"]["left_wrist_yaw_link_global_mean_cm"],
        reanchor_rows["drift10_off"]["right_wrist_yaw_link_global_mean_cm"],
    )
    on_10 = max(
        reanchor_rows["drift10_on"]["left_wrist_yaw_link_global_mean_cm"],
        reanchor_rows["drift10_on"]["right_wrist_yaw_link_global_mean_cm"],
    )
    add(
        checks,
        "mujoco_reanchor",
        set(reanchor_rows) == required_runs
        and reanchor_semantics_valid
        and reanchor["static_acceptance"]["passed"]
        and on_10 < off_10,
        {
            "static": reanchor["static_acceptance"],
            "drift10_worst_off_cm": off_10,
            "drift10_worst_on_cm": on_10,
            "experiment_semantics_valid": reanchor_semantics_valid,
        },
    )

    # Section 4: legal external references and the speed/error figure.
    sonic_manifest = yaml.safe_load(
        (TRACK / "data/sonic_hat_external/sonic_hat_manifest.yaml").read_text()
    )
    sonic_required_arrays = {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "fps",
        "focus_mask",
        "focus_time_to_next_edge_s",
        "focus_valid",
    }
    sonic_npz_evidence = []
    sonic_references_valid = True
    for name, raw_path in sonic_manifest.items():
        path = Path(raw_path)
        if not path.is_file():
            sonic_references_valid = False
            sonic_npz_evidence.append({"motion": name, "exists": False})
            continue
        with np.load(path, allow_pickle=False) as packed:
            arrays_present = sonic_required_arrays.issubset(packed.files)
            frames = int(packed["joint_pos"].shape[0]) if arrays_present else 0
            finite = arrays_present and all(
                np.isfinite(packed[key]).all()
                for key in sonic_required_arrays
                if key not in {"focus_mask", "focus_valid"}
            )
            no_focus = arrays_present and not packed["focus_valid"].any() and not packed[
                "focus_mask"
            ].any()
            quat_norm_error = (
                float(np.max(np.abs(np.linalg.norm(packed["body_quat_w"], axis=-1) - 1.0)))
                if arrays_present
                else math.inf
            )
            valid = (
                arrays_present
                and frames > 0
                and finite
                and no_focus
                and quat_norm_error < 1.0e-3
            )
            sonic_references_valid &= valid
            sonic_npz_evidence.append(
                {
                    "motion": name,
                    "exists": True,
                    "frames": frames,
                    "finite": finite,
                    "focus_valid_frames": int(packed["focus_valid"].sum()),
                    "max_quaternion_norm_error": quat_norm_error,
                    "valid": valid,
                }
            )
    sonic_comparison = load_json(RESULTS / "sonic_hat_external/comparison.json")
    external_files = (
        "released.json",
        "F3_20000.json",
        "F3h_20000.json",
        "released_vr3.json",
        "winner_vr3.json",
        "released_wholebody14.json",
        "winner_wholebody14.json",
    )
    external_reports = [load_json(RESULTS / "sonic_hat_external" / name) for name in external_files]
    expected_external_modes = [8, 8, 8, 2, 2, 7, 7]
    add(
        checks,
        "sonic_external_and_old_modes",
        len(sonic_manifest) == 2
        and sonic_references_valid
        and all(report["motions"] == 2 for report in external_reports)
        and all(report["failed_motion_count"] == 0 for report in external_reports)
        and [report["mode"]["index"] for report in external_reports]
        == expected_external_modes
        and all(
            Path(report["motion_file"]).resolve()
            == (TRACK / "data/sonic_hat_external/sonic_hat_manifest.yaml").resolve()
            for report in external_reports
        )
        and all(
            not row["all_degradations_le_10_percent"]
            for row in sonic_comparison["comparisons"]
        ),
        {
            "motions": len(sonic_manifest),
            "references": sonic_npz_evidence,
            "reports": len(external_reports),
            "mode_indices": [report["mode"]["index"] for report in external_reports],
            "hat4_le_10_claims": [
                row["all_degradations_le_10_percent"]
                for row in sonic_comparison["comparisons"]
            ],
        },
    )
    speed = load_json(RESULTS / "wrist_speed_error_10class/wrist_speed_error.json")
    speed_plot = RESULTS / "wrist_speed_error_10class/wrist_speed_error.png"
    speed_samples_path = RESULTS / "wrist_speed_error_10class/wrist_speed_error_samples.csv"
    with speed_samples_path.open(newline="") as stream:
        speed_samples = list(csv.DictReader(stream))
    sample_motions = {row["motion"] for row in speed_samples}
    sample_sides = {
        side: [row for row in speed_samples if row["side"] == side]
        for side in ("left", "right")
    }
    sample_values_finite = all(
        math.isfinite(float(row["reference_speed_deg_s"]))
        and math.isfinite(float(row["local_rotation_error_deg"]))
        for row in speed_samples
    )
    sample_means_match = all(
        math.isclose(
            float(np.mean([float(row["reference_speed_deg_s"]) for row in rows])),
            speed["per_side"][side]["reference_angular_speed_mean_deg_s"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and math.isclose(
            float(np.mean([float(row["local_rotation_error_deg"]) for row in rows])),
            speed["per_side"][side]["pelvis_local_rotation_error_mean_deg"],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for side, rows in sample_sides.items()
    )
    speed_source_reports = [
        load_json(RESULTS / "10class_3point" / f"{motion}.json")
        for motion in sorted(sample_motions)
    ]
    focus_frames_from_traces = 0
    speed_source_valid = True
    for source_report in speed_source_reports:
        trace_path = Path(source_report["trace_out"])
        reference_path = Path(source_report["reference"])
        if not trace_path.is_file() or not reference_path.is_file():
            speed_source_valid = False
            continue
        with np.load(trace_path, allow_pickle=False) as trace:
            focus_frames_from_traces += int(np.asarray(trace["focus_mask"], dtype=bool).sum())
            speed_source_valid &= (
                "active_pelvis_local_rotation_deg" in trace.files
                and "active_link_names" in trace.files
            )
    add(
        checks,
        "wrist_speed_error",
        speed["motions"] == 10
        and speed["frames_per_side"] == {"left": 2467, "right": 2467}
        and len(sample_motions) == 10
        and len(speed_samples) == 4934
        and {side: len(rows) for side, rows in sample_sides.items()}
        == {"left": 2467, "right": 2467}
        and sample_values_finite
        and sample_means_match
        and len(speed_source_reports) == 10
        and speed_source_valid
        and focus_frames_from_traces == 2467
        and sum(row["frames"] for row in speed["bins"]) == 4934
        and speed["bin_means_monotonic_non_decreasing"]
        and speed_plot.is_file()
        and speed_plot.stat().st_size > 10_000,
        {
            "motion_names": sorted(sample_motions),
            "sample_rows": len(speed_samples),
            "focus_frames_recomputed_from_traces": focus_frames_from_traces,
            "sample_means_match_json": sample_means_match,
            "ratio": speed["right_to_left_mean_speed_ratio"],
            "pooled_spearman": speed["pooled_speed_error_spearman"],
            "plot": str(speed_plot),
        },
    )

    # Official XL baseline plus the asynchronous final result.
    xl_base = TRACK / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl/model_22200.pt"
    xl_base_sha = sha256(xl_base)
    xl_baseline = load_json(RESULTS / "official_xl_hat4/released_xl_model22200.json")
    xl_baseline_failed_path = Path(xl_baseline["failed_motions_file"])
    xl_baseline_failed_count = (
        len(list(filter(None, xl_baseline_failed_path.read_text().splitlines())))
        if xl_baseline_failed_path.is_file()
        else -1
    )
    xl_baseline_metric_values = [
        float(xl_baseline["metrics"][key])
        for key in (
            "error_left_wrist_pelvis_local_pos_focus_success",
            "error_left_wrist_pelvis_local_rot_focus_success",
            "error_right_wrist_pelvis_local_pos_focus_success",
            "error_right_wrist_pelvis_local_rot_focus_success",
            "error_head_proxy_pelvis_local_pos_focus_success",
            "error_head_proxy_pelvis_local_rot_focus_success",
        )
    ]
    add(
        checks,
        "official_xl_baseline",
        xl_base_sha == "8a7be5191de8bbcd3f7359615b935ea605ee47d8703135dc4202fa9eeefc1b2f"
        and xl_baseline["simulator"] == "IsaacLab"
        and Path(xl_baseline["checkpoint"]).resolve() == xl_base.resolve()
        and Path(xl_baseline["motion_file"]).resolve() == expected_heldout_manifest
        and xl_baseline["motions"] == 600
        and xl_baseline["eval_max_steps_per_motion"] == 1000
        and xl_baseline["mode"]["index"] == 8
        and xl_baseline["mode"]["name"] == "HAT-4"
        and xl_baseline["failed_motion_count"] == 114
        and xl_baseline_failed_count == 114
        and all(
            math.isfinite(value) and value >= 0.0
            for value in xl_baseline_metric_values
        ),
        {
            "sha256": xl_base_sha,
            "passed": xl_baseline["motions"] - xl_baseline["failed_motion_count"],
            "mode": xl_baseline["mode"],
            "eval_max_steps_per_motion": xl_baseline[
                "eval_max_steps_per_motion"
            ],
            "failed_list_count": xl_baseline_failed_count,
        },
    )
    xl_run_dir = formal_root / "formal_xl_f3h_20260804"
    xl_provenance_path = xl_run_dir / "run_provenance.json"
    xl_provenance = load_json(xl_provenance_path)
    provenance_start_diff = Path(xl_provenance["launch_time_git_diff"]["path"])
    expected_xl_recipe = {
        "task": "G1-HAT4-F3H-Finetune",
        "base_seed": 42,
        "rank_seeds": [42, 43, 44, 45, 46],
        "distributed_ranks": 5,
        "environments_per_rank": 819,
        "global_environments": 4095,
        "iterations": 20000,
        "architecture": {
            "embedding_dim": 384,
            "num_heads": 6,
            "ff_dim": 384,
            "num_layers": 6,
        },
    }
    xl_provenance_valid = (
        xl_provenance["all_valid"]
        and xl_provenance["process_tree_valid"]
        and xl_provenance["detached_systemd_service_valid"]
        and len(xl_provenance["ranks"]) == 5
        and xl_provenance["warm_start_log"]["valid"]
        and xl_provenance["warm_start_log"]["official_checkpoint_line_count"] == 5
        and xl_provenance["warm_start_log"][
            "iteration_zero_and_zero_expand_line_count"
        ]
        == 5
        and xl_provenance["official_initialization"]["sha256"]
        == "8a7be5191de8bbcd3f7359615b935ea605ee47d8703135dc4202fa9eeefc1b2f"
        and xl_provenance["manifests"]["train"]["motions"] == 2400
        and xl_provenance["manifests"]["heldout"]["motions"] == 600
        and xl_provenance["manifests"]["train"]["all_references_exist"]
        and xl_provenance["manifests"]["heldout"]["all_references_exist"]
        and xl_provenance["recipe_interpretation"] == expected_xl_recipe
        and all(
            row["unchanged_before_launch_evidence"]
            for row in xl_provenance["source_snapshots"]
        )
        and not xl_provenance["launch_evidence_cutoff"]["model_0_emitted"]
        and sha256(provenance_start_diff)
        == xl_provenance["launch_time_git_diff"]["sha256"]
    )
    add(
        checks,
        "official_xl_run_provenance",
        xl_provenance_valid,
        {
            "path": str(xl_provenance_path),
            "detached_systemd_service_valid": xl_provenance[
                "detached_systemd_service_valid"
            ],
            "ranks": len(xl_provenance["ranks"]),
            "physical_gpus": sorted(
                {
                    row["environment"]["CUDA_VISIBLE_DEVICES"]
                    for row in xl_provenance["ranks"]
                }
            ),
            "warm_start_log": xl_provenance["warm_start_log"],
            "train_motions": xl_provenance["manifests"]["train"]["motions"],
            "heldout_motions": xl_provenance["manifests"]["heldout"]["motions"],
            "valid": xl_provenance_valid,
        },
    )
    xl_curve_path = RESULTS / "official_xl_hat4/checkpoints/curve_summary.json"
    if xl_curve_path.is_file():
        xl_curve = load_json(xl_curve_path)
        curve_steps = [row["step"] for row in xl_curve["rows"]]
        expected_curve_steps = [2000, 8000, 12000, 16000]
        curve_reports = []
        curve_reports_valid = True
        curve_rows_by_step = {row["step"]: row for row in xl_curve["rows"]}
        for step in expected_curve_steps:
            report_path = xl_curve_path.parent / f"model_{step}.json"
            if not report_path.is_file():
                continue
            report = load_json(report_path)
            summary_row = curve_rows_by_step.get(step, {})
            snapshot_value = summary_row.get("checkpoint_snapshot")
            snapshot = Path(snapshot_value) if snapshot_value else None
            snapshot_valid = (
                snapshot is not None
                and snapshot.is_file()
                and snapshot.stat().st_size
                == summary_row.get("checkpoint_size_bytes")
                and sha256(snapshot) == summary_row.get("checkpoint_sha256")
                and Path(report["checkpoint"]).resolve() == snapshot.resolve()
            )
            metric_values = [
                float(report["metrics"][key])
                for key in (
                    "error_left_wrist_pelvis_local_pos_focus_success",
                    "error_left_wrist_pelvis_local_rot_focus_success",
                    "error_right_wrist_pelvis_local_pos_focus_success",
                    "error_right_wrist_pelvis_local_rot_focus_success",
                    "error_head_proxy_pelvis_local_pos_focus_success",
                    "error_head_proxy_pelvis_local_rot_focus_success",
                )
            ]
            valid = (
                report["motions"] == 600
                and 0 <= report["failed_motion_count"] <= 600
                and snapshot_valid
                and all(math.isfinite(value) and value >= 0.0 for value in metric_values)
            )
            curve_reports_valid &= valid
            curve_reports.append(
                {
                    "step": step,
                    "passed": report["motions"] - report["failed_motion_count"],
                    "checkpoint_sha256": summary_row.get("checkpoint_sha256"),
                    "snapshot_valid": snapshot_valid,
                    "valid": valid,
                }
            )
        curve_complete = (
            curve_steps == expected_curve_steps
            and curve_reports_valid
            and len(curve_reports) == len(expected_curve_steps)
            and all(row.get("total") == 600 for row in xl_curve["rows"])
        )
        curve_valid_prefix = (
            curve_steps == expected_curve_steps[: len(curve_steps)]
            and curve_reports_valid
            and len(curve_reports) == len(curve_steps)
            and all(row.get("total") == 600 for row in xl_curve["rows"])
        )
        curve_passed = curve_complete or (
            args.allow_pending_xl and curve_valid_prefix
        )
        curve_evidence = {
            "steps": curve_steps,
            "reports": curve_reports,
            "complete": curve_complete,
            "valid_prefix": curve_valid_prefix,
            "path": str(xl_curve_path),
        }
    else:
        curve_passed = args.allow_pending_xl
        curve_evidence = {
            "status": "pending and explicitly allowed" if args.allow_pending_xl else "missing",
            "expected": str(xl_curve_path),
        }
    add(checks, "official_xl_checkpoint_curve", curve_passed, curve_evidence)

    xl_summary_path = TRACK / "releases/hat4_focus_release_xl_v1/final_summary.json"
    if xl_summary_path.is_file():
        xl_summary = load_json(xl_summary_path)
        xl_eval = load_json(Path(xl_summary["evaluation"]))
        released_xl = Path(xl_summary["released_checkpoint"])
        source_xl = Path(xl_summary["checkpoint"])
        xl_release_dir = xl_summary_path.parent
        xl_recipe_path = xl_release_dir / "recipe.json"
        xl_readme_path = xl_release_dir / "README.md"
        xl_recipe = load_json(xl_recipe_path) if xl_recipe_path.is_file() else {}
        metric_values = [
            float(value["candidate"])
            for value in xl_summary["fixed_hat4"]["focus_success_metrics"].values()
        ]
        xl_passed = (
            xl_eval["motions"] == 600
            and xl_summary["fixed_hat4"]["candidate_passed"]
            == 600 - xl_eval["failed_motion_count"]
            and xl_summary["fixed_hat4"]["total"] == 600
            and source_xl.is_file()
            and source_xl.stat().st_size == xl_summary["checkpoint_size_bytes"]
            and sha256(source_xl) == xl_summary["checkpoint_sha256"]
            and sha256(released_xl) == xl_summary["checkpoint_sha256"]
            and xl_readme_path.is_file()
            and xl_recipe.get("sha256") == xl_summary["checkpoint_sha256"]
            and xl_recipe.get("heldout_fixed_hat4", {}).get("total") == 600
            and xl_summary["training"] == {
                "task": "G1-HAT4-F3H-Finetune",
                "iterations": 20000,
                "distributed_ranks": 5,
                "environments_per_rank": 819,
                "global_environments": 4095,
                "rank_seeds": [42, 43, 44, 45, 46],
            }
            and all(math.isfinite(value) and value >= 0.0 for value in metric_values)
        )
        xl_evidence = {
            "passed": xl_summary["fixed_hat4"]["candidate_passed"],
            "total": xl_summary["fixed_hat4"]["total"],
            "sha256": xl_summary["checkpoint_sha256"],
        }
    else:
        xl_passed = args.allow_pending_xl
        xl_evidence = {
            "status": "pending and explicitly allowed" if args.allow_pending_xl else "missing",
            "expected": str(xl_summary_path),
        }
    add(checks, "official_xl_f3h_final", xl_passed, xl_evidence)

    main_report = ROOT / "0803-next-steps-implementation-results.md"
    failure_report = autopsy / "REPORT.md"
    paper_outline = TRACK / "paper_material/HAT4_FINETUNE_PAPER_OUTLINE.md"
    real_protocol = TRACK / "paper_material/REAL_ROBOT_REANCHOR_ACCEPTANCE_PROTOCOL.md"
    real_validator = TRACK / "scripts/analysis/validate_real_reanchor_trace.py"
    main_report_text = main_report.read_text() if main_report.is_file() else ""
    paper_outline_text = paper_outline.read_text() if paper_outline.is_file() else ""
    main_pending_markers = (
        "等待 model_20000 生成后补入",
        "XL 长跑完成后更新最终行",
        "正式长跑中",
        "最终 finetune 结果将在长跑完成后补入",
    )
    outline_pending_markers = (
        "正在运行",
        "等待自动生成",
        "待 watcher 完成",
        "XL 待补",
    )
    final_xl_sha = (
        load_json(xl_summary_path)["checkpoint_sha256"]
        if xl_summary_path.is_file()
        else None
    )
    final_docs_ready = (
        args.allow_pending_xl
        or (
            not any(marker in main_report_text for marker in main_pending_markers)
            and not any(marker in paper_outline_text for marker in outline_pending_markers)
            and "hat4_focus_release_xl_v1" in main_report_text
            and "official XL-F3h" in main_report_text
            and final_xl_sha is not None
            and final_xl_sha in main_report_text
            and "F3h-XL 20k" in paper_outline_text
        )
    )
    add(
        checks,
        "markdown_reports",
        main_report.is_file()
        and failure_report.is_file()
        and paper_outline.is_file()
        and real_protocol.is_file()
        and real_validator.is_file()
        and final_docs_ready,
        {
            "main": str(main_report),
            "failure": str(failure_report),
            "paper_outline": str(paper_outline),
            "real_reanchor_protocol": str(real_protocol),
            "real_trace_validator": str(real_validator),
            "final_xl_results_written": final_docs_ready,
        },
    )

    report = {
        "allow_pending_xl": args.allow_pending_xl,
        "all_passed": all(check["passed"] for check in checks),
        "passed": sum(check["passed"] for check in checks),
        "total": len(checks),
        "checks": checks,
        "hardware_scope": (
            "NOT LOCALLY EXECUTABLE: real-robot static calibration, online tracker "
            "re-anchor, and grasp success rate require the physical robot/tracker."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
