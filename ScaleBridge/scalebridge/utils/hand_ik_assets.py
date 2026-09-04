"""Match a body motion to its packaged Inspire fingertip IK trajectory.

Both the Sim2Sim and the real-robot backends need the hand trajectory that
belongs to the body motion being played.  Pairing them by hand invites the
silent mismatch where a pillow grasp is replayed on top of an unrelated body
motion, so resolve it from the motion filename and fail loudly when nothing
matches.
"""

import glob
import os

DEFAULT_SEARCH_DIR = (
    "/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/action_error_results"
)
_SUFFIX = "_hand_pino_ik.npz"


def _normalize(name):
    return os.path.splitext(os.path.basename(name))[0].lower()


def available_hand_ik(search_dir=DEFAULT_SEARCH_DIR):
    """Return {motion tag: absolute path} for every packaged hand IK file."""
    found = {}
    for path in sorted(glob.glob(os.path.join(search_dir, "*" + _SUFFIX))):
        found[os.path.basename(path)[: -len(_SUFFIX)].lower()] = path
    return found


def resolve_hand_ik_path(hand_ik_path, motion_path, search_dir=DEFAULT_SEARCH_DIR):
    """Turn a configured ``hand_ik_path`` into a concrete file.

    Anything other than the literal ``"auto"`` is passed through untouched, so
    an explicit override still wins.
    """
    if hand_ik_path != "auto":
        return hand_ik_path

    motion = _normalize(motion_path)
    candidates = available_hand_ik(search_dir)
    # Longest tag first: "pickup_pillow_v2" must win over "pickup_pillow".
    for tag in sorted(candidates, key=len, reverse=True):
        if tag in motion:
            return candidates[tag]

    raise FileNotFoundError(
        f"No Inspire hand IK trajectory matches motion '{os.path.basename(motion_path)}' "
        f"in {search_dir}. Available: {sorted(candidates) or 'none'}. "
        "Generate one, or set simulator.config.hand_ik_path to an explicit file."
    )
