from torch import Tensor
from typing import Dict


def focus_phase(state_buffer: Dict[str, Tensor]):
    return state_buffer["focus_phase"]
