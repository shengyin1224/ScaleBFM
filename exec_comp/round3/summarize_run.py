"""Turn one real-robot log directory into the rows the protocol asks for.

Grasp success and finger insertion are deliberately absent: neither is in the
log, and guessing them from wrist geometry would invent the very numbers the
test exists to collect. The generated CSV leaves those two columns empty for
the operator to fill in.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from exec_comp.common import log,save_json

KEYPOINTS=("root","head","left_wrist","right_wrist")


def rows(path):
    with (path/"control_steps.jsonl").open() as stream:
        for line in stream:
            line=line.strip()
            if line:yield json.loads(line)


def wrist_distance(record,side="right_wrist",target="raw_position"):
    index=KEYPOINTS.index(side)
    corrected=np.asarray(record[target])
    actual=np.asarray(record["actual_position"])
    if corrected.ndim==1:corrected=corrected.reshape(-1,3)
    if actual.ndim==1:actual=actual.reshape(-1,3)
    return float(np.linalg.norm(corrected[index]-actual[index])*100)


def summarize(path):
    path=Path(path)
    records=list(rows(path))
    if not records:raise ValueError(f"No control steps in {path}")
    enabled=any(r.get("a1_enabled",r.get("residual_cm") is not None) for r in records)
    grasp=[i for i,r in enumerate(records) if r.get("grasp_phase")]
    entries=[i for i in grasp if i==0 or not records[i-1].get("grasp_phase")]
    # The wrist error at the instant the hand starts to close is the number the
    # protocol asks for; report every entry, not only the first.
    at_close=[wrist_distance(records[i]) for i in entries]
    lookahead=[max(r.get("lookahead_cm") or [0.]) if isinstance(r.get("lookahead_cm"),list)
               else float(r.get("lookahead_cm") or 0.) for r in records]
    stalls=sum(1 for a,b in zip([0.]+lookahead[:-1],lookahead) if a<=0<b)
    fallback=[bool(r.get("fallback")) for r in records]
    switches=sum(1 for r in records if r.get("switch_event"))
    errors=np.asarray([wrist_distance(r) for r in records])
    summary={"run":str(path),"control_steps":len(records),"a1_enabled":enabled,
             "duration_s":round((records[-1]["monotonic_ns"]-records[0]["monotonic_ns"])*1e-9,2),
             "switch_events":switches,
             "grasp_entries":len(entries),"grasp_steps":len(grasp),
             "wrist_error_at_close_cm":[round(v,2) for v in at_close],
             "wrist_error_at_close_cm_mean":round(float(np.mean(at_close)),3) if at_close else None,
             "stall_events":stalls,"max_lookahead_cm":round(float(max(lookahead)),2) if lookahead else 0.,
             "fallback_steps":int(sum(fallback)),
             "fallback_fraction":round(float(np.mean(fallback)),4),
             "fallback_counters":next((r["fallback_counters"] for r in reversed(records)
                                       if "fallback_counters" in r),None),
             "tracking_error_definition":"raw K* at t versus measured actual at t; corrected-command distance is not goal tracking error",
             "tracking_error_cm":{"mean":round(float(errors.mean()),3),
                                  "p95":round(float(np.percentile(errors,95)),3)},
             "grasp_success":"operator observation: not in the log",
             "fingers_fully_inserted":"operator observation: not in the log"}
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("path",help="one log directory, or the parent with --compare")
    parser.add_argument("--compare",action="store_true",
                        help="summarize every run under the directory and group by A1 on/off")
    parser.add_argument("--csv",default=None)
    arguments=parser.parse_args()
    root=Path(arguments.path)
    if not arguments.compare:
        summary=summarize(root);log("round3_run_summary",**summary)
        save_json(root/"summary.json",summary);return
    runs=sorted(p for p in root.iterdir() if (p/"control_steps.jsonl").exists())
    if not runs:raise FileNotFoundError(f"No run directories under {root}")
    summaries=[summarize(p) for p in runs]
    destination=Path(arguments.csv or root/"protocol_table.csv")
    fields=["run","a1_enabled","grasp_success","fingers_fully_inserted",
            "wrist_error_at_close_cm_mean","stall_events","fallback_fraction",
            "switch_events","control_steps","duration_s"]
    with destination.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({**{k:summary.get(k) for k in fields},
                             "grasp_success":"","fingers_fully_inserted":""})
    groups={}
    for state in (True,False):
        selected=[s for s in summaries if s["a1_enabled"] is state]
        if not selected:continue
        closes=[v for s in selected for v in s["wrist_error_at_close_cm"]]
        groups["A1_on" if state else "A1_off"]={
            "runs":len(selected),
            "wrist_error_at_close_cm_mean":round(float(np.mean(closes)),3) if closes else None,
            "stall_events_total":int(sum(s["stall_events"] for s in selected)),
            "fallback_fraction_mean":round(float(np.mean([s["fallback_fraction"] for s in selected])),4),
            "grasp_success_rate":"N/A: fill the CSV column by observation first"}
    result={"runs":summaries,"groups":groups,"table":str(destination),
            "note":"grasp success and finger insertion are operator observations; "
                   "they are left blank on purpose and must be entered by hand"}
    save_json(root/"protocol_summary.json",result)
    log("round3_protocol_summary",groups=groups,table=str(destination))


if __name__=="__main__":main()
