"""Reuse October's exact stability implementation; add VR-3 via its Python API."""
import hashlib
import json
from pathlib import Path
import sys

TRACK = Path('/root/shengyin/ScaleBFM/ScaleTrack')
SOURCE = TRACK / 'scripts/precision_teacher_student'
sys.path.insert(0, str(SOURCE))
from eval_memory import evaluate

checkpoint, mode, jitter, output = sys.argv[1:]
reference = TRACK / 'data/hat4_unifolm_focus_v3/processed/G1_WBT_Inspire_Pick_Up_Drinks_ep_0006.npz'
report = evaluate(checkpoint, reference, mode=mode, device='cuda:0', jitter_mm=float(jitter))
report['source_sha256'] = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                         for name in ['eval_memory.py', '_legacy_mujoco.py', 'deployment_contract.json']}
report['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
report['reference_sha256'] = hashlib.sha256(reference.read_bytes()).hexdigest()
report['scope'] = 'October stability probe: one full motion plus 3 s hold, same v3 reference and phase labels'
Path(output).write_text(json.dumps(report, indent=2))
print('AUDIT_RESULT', output, flush=True)
