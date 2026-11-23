"""Read-only provenance and annotation availability audit for the 25 held-out episodes."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from exec_comp.common import ROOT,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);args=p.parse_args()
    entries=json.loads(Path(args.split).read_text())['val'];rows=[]
    for entry in entries:
        folder=Path(entry['dataset_path'])
        for filename in entry['file_list']:
            path=folder/filename
            if not path.exists():raise FileNotFoundError(path)
            with h5py.File(path) as f:
                keys=[];f.visit(keys.append)
                focus=[k for k in keys if any(t in k.lower() for t in ('focus','grasp','grip_phase','phase_label'))]
                padding=np.asarray(f['observation.state'][:,98:100])
                rows.append({'path':str(path),'type':entry['type'],'fps':float(f.attrs['fps']),
                    'frames':len(f['observation.state']),'images':{k:list(f[k].shape) for k in keys if k.startswith('observation.image.')},
                    'focus_or_grasp_annotation_keys':focus,'spare_dims_98_100_max_abs':float(np.abs(padding).max()),
                    'source':str(f.attrs.get('source',f.attrs.get('source_parquet','N/A')))})
    result={'split':args.split,'robot_clips':sum(r['type']=='robot' for r in rows),'human_clips':sum(r['type']=='human' for r in rows),
            'episodes':rows,'annotated_clips':sum(bool(r['focus_or_grasp_annotation_keys']) for r in rows)}
    save_json(ROOT/'configs/hat_dataset_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
