"""Explicit real-model factory; preserves the original selected environment."""
from pathlib import Path
import sys
_project=str(Path(__file__).resolve().parents[3])
if _project not in sys.path:sys.path.insert(0,_project)


class HAT4RealCompEnv:
    def __new__(cls,config,metadata_dict,device,execution_base_environment):
        from exec_comp.real_robot.deployment_old import environment_class
        return environment_class(execution_base_environment)(config,metadata_dict,device)
