import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TRACK = PROJECT/"ScaleTrack"
METADATA = ROOT/"assets/metadata.json"
SOURCE_CHECKPOINT_HASH = "c9c65cfb60bea8e4a254abaa8a60496e8fd0b9703751a0d70c3d322789766393"
ACTIVE = [0,7,10,13]
KP_NAMES = ["root","head","left_wrist","right_wrist"]
STATE_DIM = 64
DT = .02
HISTORY = 10
CHUNK = 100
CHUNK_FPS = 30
HEAD_OFFSET = [0.,0.,.4]


@lru_cache(maxsize=1)
def metadata():
    return json.loads(METADATA.read_text())


def checkpoint():
    paths = [TRACK/"releases/hat4_focus_release_v1/model_20000.pt",
             PROJECT/"MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000.pt"]
    return next(p for p in paths if p.exists())


def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def digest(path):
    result=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):result.update(block)
    return result.hexdigest()


def log(event,**values):
    print(json.dumps({"event":event,**values},default=str,ensure_ascii=False),flush=True)


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def np_summary(x):
    x=np.asarray(x)
    return {k:float(v) for k,v in {"mean":np.mean(x),"median":np.median(x),"p95":np.percentile(x,95)}.items()}
