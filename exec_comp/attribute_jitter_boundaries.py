"""Locate excess command acceleration relative to the actual3Hz replan instants."""
import numpy as np
from exec_comp.common import ROOT,save_json,log


def main():
    folder=ROOT/'runs/round2/a1_jitter'
    totals={'excess_near_seam':0.,'excess_total':0.,'near_count':0,'all_count':0}
    for raw in sorted((folder/'raw/telemetry').glob('clip_*.npz')):
        with np.load(raw) as r,np.load(folder/'A1/telemetry'/raw.name) as a:
            rc=r['command'][:,2:4,:3].astype(np.float64);ac=a['command'][:,2:4,:3].astype(np.float64)
            if len(rc)!=len(ac):raise ValueError('Cannot align censored trajectories')
            r2=np.linalg.norm(np.diff(rc,n=2,axis=0),axis=-1)
            a2=np.linalg.norm(np.diff(ac,n=2,axis=0),axis=-1)
            excess=np.maximum(a2-r2,0)
            replan=np.floor(np.arange(len(rc))*.02*3).astype(int)
            event=np.r_[False,np.diff(replan)!=0]
            near=np.convolve(event.astype(int),np.ones(3,dtype=int),mode='same')[1:-1]>0
            totals['excess_near_seam']+=float(excess[near].sum());totals['excess_total']+=float(excess.sum())
            totals['near_count']+=int(near.sum())*2;totals['all_count']+=excess.size
    totals['positive_excess_second_difference_fraction_near_replan']=totals['excess_near_seam']/totals['excess_total']
    totals['fraction_of_samples_near_replan']=totals['near_count']/totals['all_count']
    totals['definition']='positive excess max(norm(ddA1)-norm(ddRaw),0), within±1 control step of replan; same300 full clips'
    save_json(ROOT/'configs/jitter_boundary_attribution.json',totals);log('jitter_boundary_attribution',**totals)


if __name__=='__main__':main()
