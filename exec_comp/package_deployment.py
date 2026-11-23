"""Prepare a separate default-OFF configuration beside a copied deployment config."""
import argparse
import json
import shutil
import yaml
from exec_comp.common import ROOT,PROJECT,save_json,digest,log


def main():
    p=argparse.ArgumentParser();p.add_argument('--lowpass',action='store_true');args=p.parse_args()
    original=PROJECT/'ScaleBridge/scalebridge/config/hat4_online.yaml'
    if not original.exists():raise FileNotFoundError(original)
    directory=ROOT/'deployment/configs';directory.mkdir(parents=True,exist_ok=True)
    snapshot=directory/'hat4_online.yaml'
    if snapshot.exists():raise FileExistsError(snapshot)
    shutil.copy2(original,snapshot)
    config={'defaults':['hat4_online','_self_'],
            'hydra':{'searchpath':['file://'+str(original.parent)]},
            'exec_comp':{'enable':False,'model':str(ROOT/'releases/a1_round2/a1_correct_kp.pt'),
                         'cerebellum_model':'${agent.config.checkpoint}','residual_lowpass':args.lowpass,
                         'log_dir':str(ROOT/'runs/deployment_logs')},
            'env':{'_target_':'exec_comp.deployment_env.ExecCompOnlineEnv','config':{'exec_comp':'${exec_comp}'}}}
    destination=directory/'hat4_exec_comp.yaml';destination.write_text(yaml.safe_dump(config,sort_keys=False))
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(directory),version_base=None):
        composed=compose(config_name='hat4_exec_comp')
    resolved=OmegaConf.to_container(composed,resolve=True)
    assert resolved['exec_comp']['enable'] is False
    assert resolved['env']['config']['exec_comp']['enable'] is False
    assert resolved['env']['_target_']=='exec_comp.deployment_env.ExecCompOnlineEnv'
    report={'original_config':str(original),'original_sha256':digest(original),'copied_original':str(snapshot),'copied_sha256':digest(snapshot),
            'new_default_off_config':str(destination),'composition_checked':True,'instantiated_robot_or_sockets':False,
            'all_new_files_under_exec_comp':True,'configuration_location':'New configuration sits beside an immutable copy inside the separate package; live deployment directory is untouched.',
            'overlay_support':'Plain body tracking with original clock/RTC/overlap. FOCUS handoff/adaptive/hand-event/stall/legacy compensation overlays bypass correction and are logged until separately validated.'}
    assert report['original_sha256']==report['copied_sha256']
    save_json(ROOT/'releases/a1_round2/configuration_check.json',report);log('a1_default_off_package',**report)


if __name__=='__main__':main()
