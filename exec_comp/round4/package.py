"""Assemble a default-disabled B1 bundle; pending evidence stays explicit."""
import json
import shutil
from exec_comp.common import ROOT,SOURCE_CHECKPOINT_HASH,digest,save_json


def main(candidate=False,activation=False):
    folder=ROOT/'releases/round4'/('b1_activation_candidate' if activation else 'b1_coeffblend_candidate' if candidate else 'b1');artifact=folder/'b1_correct_kp.pt'
    if activation:
        folder.mkdir(parents=True,exist_ok=True)
        base=ROOT/'releases/round4/b1/b1_correct_kp.pt'
        if not artifact.exists():
            shutil.copy2(base,artifact);shutil.copy2(base.with_suffix('.json'),artifact.with_suffix('.json'))
        if digest(base)!=digest(artifact):raise ValueError('Activation changes runtime only; network bytes must match baseline')
    exported=json.loads(artifact.with_suffix('.json').read_text())
    if digest(artifact)!=exported['sha256']:raise ValueError('Export hash mismatch')
    if exported['frozen_cerebellum_raw_sha256']!=SOURCE_CHECKPOINT_HASH:raise ValueError('Binding mismatch')
    from exec_comp.round3.metrics import acceptance,verdict
    label='b1_lowdim_activation_v1' if activation else 'b1_lowdim_coeffblend_v1' if candidate else 'b1_lowdim'
    evaluations={}
    for population,subdir,suffix,raw in (('brain','brain_evaluation','','raw_stall-on'),
            ('motion','closed_loop','','raw_stall-on'),('dr','closed_loop','_dr','raw_dr')):
        measured=ROOT/'runs/round4'/subdir/(label+suffix)/'metrics.json'
        if not measured.exists():evaluations[population]={'status':'pending'};continue
        row=json.loads(measured.read_text())
        baseline=json.loads((ROOT/'runs/round3'/subdir/raw/'metrics.json').read_text())
        gates=verdict(acceptance(row,baseline));fraction=row['executor']['round4']['fallback_fraction']
        gates['fallback_fraction']=fraction<=(.05 if population=='dr' else .02)
        evaluations[population]=dict(status='passed' if all(v is True for v in gates.values()) else 'failed',
            gates=gates,fallback_fraction=fraction,metrics_sha256=digest(measured),metrics_file=str(measured))
    offline_passed=all(value['status']=='passed' for value in evaluations.values())
    from pathlib import Path
    deployed=Path('/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt')
    deployment_hash='24d10e69d30d253f87d65465c9fa2a0be756d67d17b3a4a08299b0084576aacf'
    if deployed.exists() and digest(deployed)!=deployment_hash:raise ValueError('Deployment checkpoint changed')
    save_json(folder/'manifest.json',dict(method='b1',artifact_sha256=digest(artifact),
        cerebellum_deployment_sha256=deployment_hash,frozen_cerebellum_raw_sha256=SOURCE_CHECKPOINT_HASH,
        training_checkpoint_sha256=exported['checkpoint_sha256'],interface='correct_kp(history H10, window100x4x9)',
        coefficient_transition=candidate,
        activation_ramp=activation,
        runtime_source_sha256={name:digest(ROOT/'round4'/name) for name in ('executor.py','coefficients.py',
            'device_history.py','script_predictor.py','deployment_adapter.py','deployment_hooks.py','deployment_env.py')},
        source_sha256={p.name:digest(p) for p in (ROOT/'round4').glob('*.py')},
        keypoints=['root','head','left_wrist','right_wrist'],enabled_by_default=False,
        offline_evaluations=evaluations,offline_all_gates_passed=offline_passed,
        acceptance_status='INCOMPLETE: see offline evaluations;4080S latency deferred by user; no real-robot execution',
        real_robot_run=False))
    import yaml
    config=yaml.safe_load((ROOT/'deployment/configs/hat4_exec_comp_round3.yaml').read_text())
    options=config['exec_comp'];options.update(method='b1',model=str(artifact),cuda_graph=False,
        coefficient_transition=candidate,
        activation_ramp=activation,
        log_dir=str(ROOT/'runs/round4/deployment_logs'))
    config['env']['_target_']='exec_comp.round4.deployment_env.ExecCompRound4Env'
    destination=ROOT/'deployment/configs'/('hat4_exec_comp_round4_activation_candidate.yaml' if activation else 'hat4_exec_comp_round4_coeff_candidate.yaml' if candidate else 'hat4_exec_comp_round4.yaml')
    destination.write_text('# Draft bundle: R4 fallback/4080S acceptance pending; enable stays false.\n'+yaml.safe_dump(config,sort_keys=False))
    # Compose only: no environment, DDS, camera or robot is instantiated.
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(destination.parent),version_base=None):
        baseline=OmegaConf.to_container(compose(config_name='hat4_exec_comp_round3'),resolve=True)
        resolved=OmegaConf.to_container(compose(config_name=destination.stem),resolve=True)
    base_env=dict(baseline['env']['config']);new_env=dict(resolved['env']['config'])
    base_env.pop('exec_comp');new_env.pop('exec_comp')
    checks=dict(disabled=resolved['exec_comp']['enable'] is False,
        method_b1=resolved['exec_comp']['method']=='b1',
        env_switch_disabled=resolved['env']['config']['exec_comp']['enable'] is False,
        original_env_options_equal=base_env==new_env,
        correct_target=resolved['env']['_target_']=='exec_comp.round4.deployment_env.ExecCompRound4Env')
    if not all(checks.values()):raise AssertionError(checks)
    save_json(folder/'config_validation.json',dict(checks=checks,environment_instantiated=False,real_robot_run=False))
    shutil.copy2(ROOT/'round4/REAL_ROBOT_PROTOCOL.md',ROOT/'releases/round4/REAL_ROBOT_PROTOCOL.md')
    save_json(folder/'status.json',dict(package_files_created=True,enable=False,
        ready_for_robot=False,offline_evaluations=evaluations,offline_all_gates_passed=offline_passed,
        pending=[k+':'+v['status'] for k,v in evaluations.items() if v['status']!='passed']+['4080S latency deferred until user notifies idle window'],
        a1_lowdim_training='skipped per user',caps='deferred per user'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--coefficient-candidate',action='store_true');group.add_argument('--activation-candidate',action='store_true')
    args=parser.parse_args();main(args.coefficient_candidate,args.activation_candidate)
