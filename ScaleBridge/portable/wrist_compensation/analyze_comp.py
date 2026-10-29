import json, sys, glob, os
import numpy as np

def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def stats(recs, focus_only):
    sel = [r for r in recs if "meas_lw" in r and (not focus_only or r["focus"] >= 0.5)]
    if not sel:
        return None
    def err(a, b):
        return np.linalg.norm(np.array([r[a] for r in sel]) - np.array([r[b] for r in sel]), axis=1)
    # global wrist error vs the ORIGINAL (uncompensated) world target
    g_lw, g_rw = err("meas_lw", "ref_lw"), err("meas_rw", "ref_rw")
    # pelvis-local proxy (position only): (wrist - root)_meas vs (wrist - root)_ref
    def local(a, b, ra, rb):
        m = np.array([r[a] for r in sel]) - np.array([r[ra] for r in sel])
        t = np.array([r[b] for r in sel]) - np.array([r[rb] for r in sel])
        return np.linalg.norm(m - t, axis=1)
    l_lw = local("meas_lw", "ref_lw", "meas_root_now", "ref_root")
    l_rw = local("meas_rw", "ref_rw", "meas_root_now", "ref_root")
    root = err("meas_root_now", "ref_root")
    z = np.array([r["meas_root_now"][2] for r in sel])
    off = np.linalg.norm(np.array([r["applied_offset"] for r in sel]), axis=1)
    alpha = np.array([r["alpha"] for r in sel])
    return dict(n=len(sel),
                g_lw=g_lw.mean(), g_rw=g_rw.mean(), g_rw_p95=np.percentile(g_rw, 95),
                l_lw=l_lw.mean(), l_rw=l_rw.mean(),
                root=root.mean(), zmin=z.min(), off=off.mean(), amax=alpha.max())

print(f"{'mode':13s} {'win':6s} {'n':>4s} | {'gL':>5s} {'gR':>5s} {'gRp95':>6s} | {'locL':>5s} {'locR':>5s} | {'root':>5s} {'zmin':>5s} | {'off':>5s} {'amax':>4s}   (cm)")
for d in sys.argv[1:]:
    path = os.path.join(d, "wrist_compensation_trace.jsonl")
    recs = load(path)
    mode = recs[len(recs)//2]["mode"]
    for focus_only, name in ((True, "FOCUS"), (False, "all")):
        s = stats(recs, focus_only)
        if s is None:
            continue
        print(f"{mode:13s} {name:6s} {s['n']:4d} | "
              f"{s['g_lw']*100:5.2f} {s['g_rw']*100:5.2f} {s['g_rw_p95']*100:6.2f} | "
              f"{s['l_lw']*100:5.2f} {s['l_rw']*100:5.2f} | "
              f"{s['root']*100:5.2f} {s['zmin']:5.2f} | {s['off']*100:5.2f} {s['amax']:4.2f}")
