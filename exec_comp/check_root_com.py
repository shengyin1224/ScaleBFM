"""Audit simulator pelvis COM offset without collecting training data."""
from contextlib import ExitStack
from isaaclab.app import AppLauncher
import argparse
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app
import torch
from isaaclab.sim import SimulationContext
from exec_comp.common import ROOT,TRACK,import_path,save_json,log
from exec_comp.env import make_env
lifecycle=import_path("exec_comp_com_lifecycle",TRACK/"scripts/precision_teacher_student/student_lifecycle.py")


def main(resources):
    env=make_env(1,"validation",args.device);resources.callback(env.close)
    offset=env.unwrapped.scene["robot"].data.body_com_pos_b[0,0].tolist()
    result={"root_com_offset_b":offset,"root_lin_vel_b_is_com_velocity":True,"root_position_is_link_origin":True}
    save_json(ROOT/"configs/root_com_contract.json",result);log("root_com_contract",**result)


if __name__=="__main__":
    resources=ExitStack();lifecycle.run_with_cleanup(lambda:main(resources),app,resources,SimulationContext.instance)
