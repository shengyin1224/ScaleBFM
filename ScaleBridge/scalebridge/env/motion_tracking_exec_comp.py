"""Explicit opt-in entry; the original entry and environment stay untouched.

Only Hydra +exec_comp=a1|b1 imports this file. Bootstrap the sibling exec_comp
package for the user's unchanged `python scalebridge/run.py` launch form;
no global PYTHONPATH or site-packages modifications are needed.
"""
from pathlib import Path
import sys

_project = str(Path(__file__).resolve().parents[3])
if _project not in sys.path:
    sys.path.insert(0, _project)

from exec_comp.round4.deployment_env import environment_class


class HAT4ExecCompEnv:
    """Use Hydra's original env choice, never infer it from the FOCUS flag."""
    def __new__(cls,config,metadata_dict,device,execution_base_environment):
        env_type=environment_class(execution_base_environment)
        return env_type(config,metadata_dict,device)
