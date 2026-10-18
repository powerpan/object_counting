"""S-only class-crossfit fusion, permanently excluded from final candidacy."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.fusion import fit_gate,gate_weights
from planb.round2.spatial import fused_map,cell_mass,game_from_cells
from planb.density import boxes_to_density
from planb.round2.common import sha,write_json
from m4_v2_common import settings,begin,deadline,fail,read
from round2_m4_validate import summarize
from audit_b0_results import read_rows
from analyze_m2_m3 import paired_intervals


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--time-limit-seconds',type=float,required=True)
    a=p.parse_args();root=a.root.resolve();base,old,c,splits=settings(root);out,start=begin(root,'s-fusion-record',a.time_limit_seconds,1800)
    try:
        source=root/'results/round2-v1/val-head-S-seed42';assert read(source/'audit.json')['status']=='passed'
        rows={r['image']:r for r in read_rows(source/'predictions.jsonl')}
        baseline={r['image']:r for r in read_rows(root/base['baseline_result']/'predictions.jsonl')}
        labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
        protocol=read(out/'protocol.json');protocol['S_evidence_sha256']={n:sha(source/n) for n in ('predictions.jsonl','summary.json','audit.json')}
        write_json(out/'protocol.json',protocol);folds={};predictions={}
        for f,t in [('A','B'),('B','A')]:
            deadline(start,a.time_limit_seconds);fit=splits['val_'+f];assess=splits['val_'+t]
            assert set(labels[n] for n in fit).isdisjoint(labels[n] for n in assess)
            r=[rows[n] for n in fit]
            model=fit_gate([x['features'] for x in r],[x['m1_cell_mass'] for x in r],
                           [x['density_cell_mass'] for x in r],[x['truth_cell_count'] for x in r],base['m4']['fusion'])
            folds[f]={'fit_images':fit,'assess_images':assess,'gate':model,'fixed_width':'S'}
            for name in assess:
                deadline(start,a.time_limit_seconds);r=rows[name];b=baseline[name];weights=gate_weights(r['features'],model)
                path=root/'work/round2-v1/val-head-S-seed42/maps'/(Path(name).stem+'.npz');assert sha(path)==r['density_map_sha256']
                with np.load(path) as z:d=z['density']
                m1=boxes_to_density(b['boxes_xyxy'],b['height'],b['width']);combined=fused_map(m1,d,weights)
                cells=cell_mass(combined);expected=weights*np.asarray(r['m1_cell_mass'])+(1-weights)*np.asarray(r['density_cell_mass'])
                np.testing.assert_allclose(cells,expected,atol=1e-9,rtol=1e-12)
                predictions[name]={'image':name,'truth':b['truth'],'prediction':float(combined.sum()),
                    'game1':game_from_cells(cells,r['truth_cell_count'],1),'game2':game_from_cells(cells,r['truth_cell_count'],2),
                    'weights':weights.tolist(),'cell_mass':cells.tolist(),'fit_fold':f,'assess_fold':t}
        ordered=[predictions[n] for n in baseline];write_json(out/'folds.json',folds)
        with (out/'predictions.jsonl').open('x') as stream:
            for r in ordered:stream.write(json.dumps(r)+'\n')
        write_json(out/'summary.json',{'status':'complete',**summarize(ordered),'candidate_eligible':False,
            'purpose':'fixed S-only historical record; never final candidate','width_selection_performed':False,
            'paired_vs_b0':{u:paired_intervals(ordered,baseline,labels,2000,u) for u in ('image','class')},
            'elapsed_seconds':time.monotonic()-start})
    except BaseException as e:fail(out,start,e);raise


if __name__=='__main__':main()
