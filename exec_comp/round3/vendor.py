"""Hash-pinned, byte-identical snapshot of the deployment switching modules.

The training host carries an older partial ScaleBridge checkout, so the
simulation cannot import the live deployment package there. Copying is the
only operation performed: no deployment file is edited, and every copy is
recorded with the sha256 of its source. ``verify`` refuses to run when a
vendored file has drifted from the still-present original, so a silent fork
between what the robot runs and what the simulation reuses cannot survive.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
from exec_comp.common import ROOT,PROJECT,digest,save_json,log

VENDOR=ROOT/"vendor"
BRIDGE=PROJECT/"ScaleBridge"
MANIFEST=VENDOR/"VENDOR_MANIFEST.json"
# Exactly the modules Python loads when importing the deployed adaptive
# environment, plus the hand-event guard the config can switch on.
MODULES=(
    "scalebridge/__init__.py",
    "scalebridge/env/__init__.py",
    "scalebridge/env/base_env.py",
    "scalebridge/env/motion_tracking_hat4_online.py",
    "scalebridge/env/motion_tracking_hat4_adaptive.py",
    "scalebridge/online/__init__.py",
    "scalebridge/online/adaptive_chunk_execution.py",
    "scalebridge/online/continuity_hand_event_guard.py",
    "scalebridge/online/current_pose.py",
    "scalebridge/online/focus.py",
    "scalebridge/online/focus_handoff.py",
    "scalebridge/online/focus_worker.py",
    "scalebridge/online/gpt_supervisor.py",
    "scalebridge/online/protocol.py",
    "scalebridge/online/reference.py",
    "scalebridge/online/stall_wrist_assist.py",
    "scalebridge/online/trajectory_continuity.py",
    "scalebridge/online/transport.py",
)


def update():
    if not BRIDGE.exists():raise FileNotFoundError(f"Deployment source tree missing: {BRIDGE}")
    files={}
    for relative in MODULES:
        source=BRIDGE/relative;destination=VENDOR/relative
        if not source.exists():raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        files[relative]={"source":str(source),"sha256":digest(source),"bytes":source.stat().st_size}
    save_json(MANIFEST,{"source_tree":str(BRIDGE),"files":files,
        "policy":"byte-identical copies; the originals are never written to",
        "reason":"the training host has no live ScaleBridge checkout, and the simulation must reuse "
                 "the deployed switching rules rather than a reimplementation"})
    log("vendor_updated",files=len(files))
    return files


def verify(strict_against_source=None):
    """Check the snapshot; compare with the originals wherever they exist."""
    if not MANIFEST.exists():raise FileNotFoundError(f"Run `python -m exec_comp.round3.vendor update` first: {MANIFEST}")
    manifest=json.loads(MANIFEST.read_text())
    # A partial checkout must not be reported as a source comparison: the
    # training host keeps an old ScaleBridge tree where most of these modules
    # are simply absent, and "no mismatch found" there means nothing.
    present=[r for r in manifest["files"] if (BRIDGE/r).exists()]
    complete=len(present)==len(manifest["files"])
    if strict_against_source and not complete:
        raise RuntimeError(f"Only {len(present)}/{len(manifest['files'])} originals present under {BRIDGE}")
    drift=[]
    for relative,record in manifest["files"].items():
        copy=VENDOR/relative
        if not copy.exists():drift.append({"file":relative,"problem":"vendored copy missing"});continue
        if digest(copy)!=record["sha256"]:drift.append({"file":relative,"problem":"vendored copy edited"})
        source=BRIDGE/relative
        if source.exists() and digest(source)!=record["sha256"]:
            drift.append({"file":relative,"problem":"deployment source changed since vendoring"})
    if drift:raise RuntimeError(f"Vendored switching code is out of date: {drift}")
    return {"files":len(manifest["files"]),"compared_with_source":len(present),
            "source_tree_complete":complete,"source_tree":manifest["source_tree"]}


def activate():
    """Put the pinned snapshot on sys.path and return its provenance."""
    state=verify()
    path=str(VENDOR)
    if path not in sys.path:sys.path.insert(0,path)
    import scalebridge
    if not Path(scalebridge.__file__).is_relative_to(VENDOR):
        raise RuntimeError(f"A different scalebridge package is already imported: {scalebridge.__file__}")
    return state


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("update","verify"))
    arguments=parser.parse_args()
    if arguments.action=="update":update()
    log("vendor_"+arguments.action,**verify())
