"""Bounded deployment checks; no training, model selection, or full benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sources(root):
    receipt = read(root / 'provenance/hpc4_source_receipt_v1.json')
    if not receipt['remote_main_verified']:
        raise ValueError('Migration entry must be committed and pushed first')
    for name, expected in receipt['files_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root / relative) != expected:
            raise ValueError('Source mismatch: ' + name)
    return receipt


def asset_records(root):
    cfg = read(root / 'configs/hpc4_migration_v1.json')
    final = read(root / 'configs/final_v1.json')
    manifest = read(root / 'manifests/fsc147_val.json')
    records = [
        {'path': 'weights/GECO2FSCD.pth', 'sha256': final['geco_sha256'],
         'url': 'https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/'
                'ed3c8ff3753e731fd7074862c0ea49d908785335/weights/GECO2FSCD.pth'},
        {'path': 'cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt',
         'sha256': final['sam2_sha256'],
         'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt'},
    ]
    for item in manifest['files']:
        path = Path(item['path'])
        if item['kind'] == 'image' and path.name not in cfg['val_images']:
            continue
        record = dict(item)
        record['url'] = (f"https://huggingface.co/datasets/isentropic/FSC147/resolve/"
                         f"{manifest['mirror_revision']}/images_384_VarV2/{path.name}"
                         if item['kind'] == 'image' else item['source'])
        records.append(record)
    if sum(item.get('kind') == 'image' for item in records) != len(cfg['val_images']):
        raise ValueError('Missing fixed validation image')
    return records


def assets(root):
    out = root / 'provenance/hpc4_assets_v1.json'
    if out.exists():
        raise FileExistsError('Asset receipt already exists; inspect it instead of rerunning')
    checked = []
    for item in asset_records(root):
        dest = root / item['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            partial = dest.with_suffix(dest.suffix + '.partial')
            with partial.open('xb') as stream:
                with urllib.request.urlopen(item['url'], timeout=45) as response:
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
            if sha(partial) != item['sha256']:
                raise ValueError('Downloaded asset hash mismatch: ' + item['path'])
            partial.replace(dest)
        if sha(dest) != item['sha256']:
            raise ValueError('Existing asset hash mismatch: ' + item['path'])
        if 'bytes' in item and dest.stat().st_size != item['bytes']:
            raise ValueError('Asset size mismatch: ' + item['path'])
        checked.append({'path': item['path'], 'sha256': item['sha256'], 'bytes': dest.stat().st_size})
        print(json.dumps({'asset_verified': item['path']}), flush=True)
    write_new(out, {'status': 'passed', 'files': checked, 'download_location': 'school server'})


def smoke(root, receipt):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('An allocated Slurm GPU is required')
    cfg = read(root / 'configs/hpc4_migration_v1.json')
    final = read(root / 'configs/final_v1.json')
    method = read(root / final['method_config'])
    if sha(root / final['method_config']) != final['method_config_sha256']:
        raise ValueError('Frozen method changed')
    for item in asset_records(root):
        if sha(root / item['path']) != item['sha256']:
            raise ValueError('Asset changed after download')
    upstream = root / 'external/GECO2'
    pin = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != final['upstream_commit']:
        raise ValueError('Upstream revision differs')
    for args in (['diff', '--quiet'], ['diff', '--cached', '--quiet']):
        subprocess.run(['git', '-C', str(upstream), *args], check=True)
    output = root / 'results/hpc4-migration-v1'
    output.mkdir(parents=True, exist_ok=False)
    os.environ['TORCH_HOME'] = str(root / 'cache/torch')
    sys.path[:0] = [str(upstream), str(root), str(root / 'scripts')]
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, str(root / 'scripts/check_cuda_ops.py'),
                        '--upstream', str(upstream), '--output', str(output / 'cuda_ops.json')],
                       check=True, timeout=120)
        from planb.round2.common import seed_everything
        from run_m2_m3 import load_model
        from planb.final_inference import FinalEngine
        seed_everything(42)
        torch.set_num_threads(4)
        model = load_model(root)
        engine = FinalEngine(model, method)
        annotations = read(root / 'data/fsc147/annotations/annotation_FSC147_384.json')
        split = read(root / 'data/fsc147/annotations/Train_Test_Val_FSC_147.json')
        rows = []
        for name in cfg['val_images']:
            if name not in split['val'] or name in split['train'] + split['test']:
                raise ValueError('Smoke image is not exclusively validation')
            image = Image.open(root / 'data/fsc147/images_384_VarV2' / name).convert('RGB')
            anno = annotations[name]
            prompts = torch.tensor(anno['box_examples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
            b0, result, bd, fd, detail = engine.run_pair(T.ToTensor()(image), prompts, gate=True)
            expected = cfg['b0_counts'][name]
            if b0['prediction'] != expected:
                raise ValueError('B0 count differs from saved AutoDL result: ' + name)
            for entry, density in ((b0, bd), (result, fd)):
                if not np.isfinite(density).all() or not np.isclose(density.sum(), entry['prediction'], rtol=0, atol=1e-6):
                    raise ValueError('Invalid density or lost mass')
                if entry['peak_allocated_mib'] > cfg['max_allocated_mib']:
                    raise RuntimeError('Deployment memory limit exceeded')
            rows.append({'image': name, 'baseline_count': b0['prediction'],
                         'baseline_matches_saved_result': True, 'final_count': result['prediction'],
                         'route': detail['route'],
                         'peak_allocated_mib': max(b0['peak_allocated_mib'], result['peak_allocated_mib']),
                         'final_inference_seconds': result['inference_seconds']})
            print(json.dumps(rows[-1]), flush=True)
        write_new(output / 'smoke.json', {'status': 'passed', 'code_commit': receipt['code_commit'],
                  'gpu': torch.cuda.get_device_name(0),
                  'total_memory_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'elapsed_seconds': time.monotonic() - started, 'rows': rows,
                  'scope': 'Deployment smoke only; not a new accuracy or generalization experiment'})
    except BaseException as exc:
        write_new(output / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                  'message': str(exc), 'elapsed_seconds': time.monotonic() - started,
                  'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['sources', 'assets', 'smoke'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = sources(root)
    if args.phase == 'sources':
        print(json.dumps({'source_check': 'passed', 'files': len(receipt['files_sha256'])}))
    elif args.phase == 'assets':
        assets(root)
    else:
        smoke(root, receipt)


if __name__ == '__main__':
    main()
