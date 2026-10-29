"""Opt-in factory for +exec_comp=real_capture; preserve original env selection."""
from pathlib import Path
import sys

_project = str(Path(__file__).resolve().parents[3])
if _project not in sys.path:
    sys.path.insert(0, _project)


class HAT4RealCaptureEnv:
    def __new__(cls, config, metadata_dict, device, execution_base_environment):
        from exec_comp.real_robot.capture import environment_class
        return environment_class(execution_base_environment)(config, metadata_dict, device)
