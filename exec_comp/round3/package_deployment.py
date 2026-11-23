"""A1-6 item 1: a default-off round-3 config beside an untouched copy.

The live deployment directory is never written to. The round-3 config layers
``exec_comp`` on top of byte-identical copies of the two files the robot runs
(the top-level entry point and the adaptive environment), and the composition
is checked offline: no robot, no socket, no simulator is instantiated.
"""
import argparse
import json
import shutil
import yaml
from exec_comp.common import ROOT,PROJECT,save_json,digest,log

CONFIG=PROJECT/"ScaleBridge/scalebridge/config"
DIRECTORY=ROOT/"deployment/configs"
# The command the robot was last run with, from the recorded hydra overrides.
DEPLOYED_OVERRIDES={
    "env":"motion_tracking_hat4_adaptive",
    "env.config.reference_forcing":False,
    "env.config.min_chunk_execution_fraction":0.1,
    "env.config.chunk_start_reference_continuity":True,
    "env.config.focus_handoff_enabled":True,
    "env.config.focus_handoff_version":"v3",
    "env.config.focus_handoff_source":"keypoints",
    "env.config.focus_handoff_finger_mode":"hold_progress",
}


def copy_once(source,destination):
    if destination.exists():
        if digest(destination)!=digest(source):
            raise FileExistsError(f"{destination} differs from {source}; refusing to overwrite")
        return destination
    shutil.copy2(source,destination);return destination


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("stepwise","switch_event"),default="stepwise")
    parser.add_argument("--release",default=str(ROOT/"releases/a1_round3/a1_correct_kp.pt"))
    arguments=parser.parse_args()
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    (DIRECTORY/"env").mkdir(exist_ok=True)
    originals={
        "hat4_online.yaml":CONFIG/"hat4_online.yaml",
        "env/motion_tracking_hat4_adaptive.yaml":CONFIG/"env/motion_tracking_hat4_adaptive.yaml",
        "env/motion_tracking_hat4_online.yaml":CONFIG/"env/motion_tracking_hat4_online.yaml",
    }
    copies={}
    for relative,source in originals.items():
        if not source.exists():raise FileNotFoundError(source)
        copies[relative]=copy_once(source,DIRECTORY/relative)
    release=json.loads((ROOT/"releases/a1_round3/manifest.json").read_text())
    config={
        # hat4_online already selects an env group, so this is an override,
        # exactly as the recorded launch command's `env=...` is.
        "defaults":["hat4_online",{"override /env":"motion_tracking_hat4_adaptive"},"_self_"],
        "hydra":{"searchpath":["file://"+str(CONFIG)]},
        "exec_comp":{
            "enable":False,
            "log_baseline":True,
            "model":arguments.release,
            "cerebellum_model":"${agent.config.checkpoint}",
            "mode":arguments.mode,
            "residual_lowpass":release["residual_lowpass_50hz"],
            "total_offset_cap_m":release["shared_cap"]["total_offset_cap_m"],
            "grasp_release_ramp_s":release["grasp_behaviour"]["release_ramp_s"],
            "seam_crossfade_s":release["call_mode"]["seam_crossfade_s"],
            "fallback_ramp_s":release["safety_fallback"]["ramp_s"],
            "cap_hold_s":release["safety_fallback"]["sustained_cap_s"],
            "cap_guard":release["safety_fallback"]["cap_guard_reading"],
            'keep_lookahead':release.get('lookahead_policy',{}).get('keep',True),
            "max_step_cm":release["safety_fallback"]["single_step_jump_cm"],
            "log_dir":str(ROOT/"runs/round3/deployment_logs"),
        },
        "env":{"_target_":"exec_comp.round3.deployment_env.ExecCompRound3Env",
               "config":{"exec_comp":"${exec_comp}",**{k.split("env.config.")[1]:v
                         for k,v in DEPLOYED_OVERRIDES.items() if k.startswith("env.config.")}}},
    }
    destination=DIRECTORY/"hat4_exec_comp_round3.yaml"
    destination.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    alias=DIRECTORY/'hat4_exec_comp.yaml'
    alias_text=destination.read_text()  # Hydra searchpath must be present in the primary config
    if alias.exists() and alias.read_text()!=alias_text:
        archive=ROOT/'deployment/round2/configs/hat4_exec_comp.yaml'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():shutil.copy2(alias,archive)
    alias.write_text(alias_text)
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(DIRECTORY),version_base=None):
        composed=compose(config_name="hat4_exec_comp_round3")
        alias_composed=compose(config_name="hat4_exec_comp")
    resolved=OmegaConf.to_container(composed,resolve=True)
    alias_resolved=OmegaConf.to_container(alias_composed,resolve=True)
    checks={
        'canonical_config_alias_matches_round3':alias_resolved==resolved,
        'baseline_logging_readonly':resolved['exec_comp']['log_baseline'] is True,
        "exec_comp_disabled_by_default":resolved["exec_comp"]["enable"] is False,
        "env_sees_the_same_switch":resolved["env"]["config"]["exec_comp"]["enable"] is False,
        "target_is_the_round3_env":resolved["env"]["_target_"]=="exec_comp.round3.deployment_env.ExecCompRound3Env",
        "wrist_compensation_is_none":resolved["env"]["config"]["wrist_compensation"]=="none",
        "adaptive_switching_preserved":all(
            resolved["env"]["config"][key.split("env.config.")[1]]==value
            for key,value in DEPLOYED_OVERRIDES.items() if key.startswith("env.config.")),
        "call_mode_present":resolved["exec_comp"]["mode"] in ("stepwise","switch_event"),
        "shared_cap_present":resolved["exec_comp"]["total_offset_cap_m"]==0.05,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise AssertionError(f"round-3 configuration check failed: {failed}")
    report={"new_config":str(destination),"checks":checks,
            "copied_originals":{k:{"source":str(v),"sha256":digest(v),
                                   "copy":str(copies[k]),"copy_sha256":digest(copies[k])}
                                for k,v in originals.items()},
            "originals_unmodified":all(digest(v)==digest(copies[k]) for k,v in originals.items()),
            "deployed_overrides_reproduced":DEPLOYED_OVERRIDES,
            "instantiated_robot_or_sockets":False,
            "mutual_exclusion":("exec_comp.enable with wrist_compensation != none raises before "
                "the environment is built; see exec_comp/round3/deployment_env.py"),
            "grasp_overlays":("FOCUS handoff, hand-event guard, stall assist and the predicted "
                "close boundary freeze the residual and ramp back over 150 ms; round 2 bypassed "
                "A1 in those states, which dropped the residual to zero in one control step"),
            "real_robot_run":False}
    save_json(ROOT/"releases/a1_round3/configuration_check.json",report)
    log("a1_round3_package",**{k:report[k] for k in ("new_config","originals_unmodified","real_robot_run")})


if __name__=="__main__":main()
