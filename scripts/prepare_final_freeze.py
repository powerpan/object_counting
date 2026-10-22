"""Create the preregistered test manifest and val-only size cutoffs; no inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from PIL import Image


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main(root, assets):
    ann_dir = assets/'data/fsc147/annotations'
    ann = json.loads((ann_dir/'annotation_FSC147_384.json').read_text())
    splits = json.loads((ann_dir/'Train_Test_Val_FSC_147.json').read_text())
    assert len(splits['test']) == 1190 and len(splits['val']) == 1286
    assert not set(splits['test']) & set(splits['train']+splits['val'])
    labels = dict(x.split(maxsplit=1) for x in (ann_dir/'ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    assert len({labels[n] for n in splits['test']}) == 29
    base = [json.loads(x) for x in (root/'results/b0-val-b1-boundary-r1/predictions.jsonl').read_text().splitlines()]
    assert [r['image'] for r in base] == splits['val']
    sizes = []
    for row in base:
        e = np.asarray(ann[row['image']]['box_examples_coordinates'], dtype=float)[:3, [0, 2], :].reshape(3, 4)
        area = np.prod(e[:, 2:]-e[:, :2], axis=1)
        assert (area > 0).all()
        sizes.append(float(np.sqrt(area.mean()/(row['width']*row['height']))))
    cut = np.quantile(sizes, [1/3, 2/3], method='linear').tolist()
    files = []
    with zipfile.ZipFile(assets/'data/fsc147/images.zip') as z:
        for name in splits['test']:
            raw = z.read('images_384_VarV2/'+name)
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
            files.append({'image':name, 'path':'data/fsc147/images_384_VarV2/'+name,
                          'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw), 'width':w, 'height':h})
    manifest = {'split':'test','n':1190,'class_count':29,'ids':splits['test'],
                'source':'original official FSC-147 images_384_VarV2 archive',
                'source_archive_sha256':sha(assets/'data/fsc147/images.zip'),
                'annotation_sha256':{p.name:sha(p) for p in ann_dir.iterdir() if p.is_file()}, 'files':files}
    write(root/'manifests/fsc147_test_v1.json', manifest)
    cp = json.loads((root/'results/b0-val-b1-boundary-r1/checkpoint_load.json').read_text())
    rng = np.random.default_rng(20261022)
    random_names = rng.choice(splits['test'], size=6, replace=False).tolist()
    cfg = {'version':'final-v1','method_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
           'upstream_commit':'67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b',
           'method_config':'configs/round3_v1.json','method_config_sha256':sha(root/'configs/round3_v1.json'),
           'method':'M3-v2-g3 when triggered, otherwise T2', 'baseline':'B0 original view',
           'mask':'automatic; tile branch follows original view', 'calibration':False, 'm2':False,
           'precision':'FP32','tf32':False,'seed':42,'batch_size':1,
           'trigger':{'median_sqrt_exemplar_area_model_pixels_le':32,'b0_count_ge':200,'operator':'or'},
           'tiling':{'grid':3,'halo_fraction':.1,'threshold':'global maximum across tiles * 0.11','nms_iou':.5},
           'tta':{'views':['original','horizontal_flip'],'count':'arithmetic mean','density':'mean of original-grid M1 maps'},
           'localization':{'boxes':'M3 merged boxes for triggered images; original-view boxes for T2',
                           'radius':'0.5 * mean exemplar width and height','matching':'maximum cardinality bipartite'},
           'size_groups':{'formula':'sqrt(mean(first_three_exemplar_areas)/(image_width*image_height))',
                          'source':'all 1286 official val images','quantile_method':'linear','cutoffs':cut,
                          'boundaries':['small <= lower','lower < medium <= upper','large > upper']},
           'density_groups':['<50','50-299','>=300'],'game_levels':[1,2],
           'bootstrap':{'replicates':2000,'seed':42,'units':['image','class'],'method':'paired multinomial percentile 95%'},
           'success':{'rmse_ratio_max':.98,'rmse_image_ci_upper_strict_max':0,'dense_mae_ratio_max':.95,'mae_ratio_max':1.02},
           'test_manifest':'manifests/fsc147_test_v1.json','test_manifest_sha256':sha(root/'manifests/fsc147_test_v1.json'),
           'geco_sha256':cp['fsc_sha256'],'sam2_sha256':cp['sam2_sha256'],
           'human_blind_labels':{'status':'omitted_by_owner_amendment','kappa':None,'occlusion_clutter_claims':False},
           'qualitative':{'random_seed':20261022,'random_six':random_names,'worst_final':6,'improved':3,'regressed':3,
                          'ties':'image filename ascending','overlap':'retain selection reasons; render duplicates once',
                          'mechanism_val':['865.jpg','3665.jpg']},
           'budgets_seconds':{'test_and_carpk':7200,'preflight':1800,'demo':3600},
           'max_memory_allocated_mib':40000,'test_retries':0,
           'carpk':{'split':'official test','exemplars':'first three annotation boxes in original file order','tuning':False},
           'demo':{'source':'val only; no rerun of test images','dense_val':next(r['image'] for r in base if r['truth']>=300),
                   'ordinary_val':next(r['image'] for r in base if r['truth']<50)}}
    write(root/'configs/final_v1.json', cfg)
    print(json.dumps({'test_images':1190,'size_cutoffs':cut,'config_sha256':sha(root/'configs/final_v1.json'),
                      'manifest_sha256':cfg['test_manifest_sha256']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.assets.resolve())
