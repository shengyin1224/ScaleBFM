"""Explicit5cm four-experiment entry, preserving the original env choice."""
from pathlib import Path
import sys
_project=str(Path(__file__).resolve().parents[3])
if _project not in sys.path:sys.path.insert(0,_project)


class HAT4FiveCompEnv:
    def __new__(cls,config,metadata_dict,device,execution_base_environment):
        from exec_comp.real_robot.deployment_four import environment_class
        return environment_class(execution_base_environment)(config,metadata_dict,device)
