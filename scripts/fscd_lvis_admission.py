"""CPU-only full-pass admission after the frozen smoke audit."""
import argparse
import math
import os
from pathlib import Path
from fscd_lvis_common import checked_sources, read, sha, write_new


def admission_seconds(smoke_seconds, smoke_images, full_images):
    if (not math.isfinite(smoke_seconds) or smoke_seconds <= 0
            or not isinstance(smoke_images, int) or smoke_images <= 0
            or not isinstance(full_images, int) or full_images <= 0):
        raise ValueError('Invalid runtime evidence or scope')
    return 1.5 * smoke_seconds / smoke_images * full_images + 300


def admit(root):
    if os.environ.get('SLURM_JOB_PARTITION') != 'amd' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Admission must run without a GPU')
    entry = checked_sources(root, 'fscd_lvis_pipeline_receipt_v1.json')
    inference = checked_sources(root, 'fscd_lvis_eval_receipt_v1.json')
    checked_sources(root, 'fscd_lvis_freeze_receipt_v1.json')
    folder = root/'results/fscd-lvis-external-v1/smoke'
    summary = read(folder/'summary.json')
    audit = read(folder/'audit.json')
    frozen = read(root/'configs/fscd_lvis_manifest_freeze_v1.json')
    cfg = read(root/'configs/fscd_lvis_external_v1.json')
    if (summary['status'] != 'complete' or audit['status'] != 'passed'
            or summary['source_commit'] != inference['code_commit']
            or audit['source_commit'] != inference['code_commit']
            or audit['summary_sha256'] != sha(folder/'summary.json')
            or audit['predictions_sha256'] != sha(folder/'predictions.jsonl')
            or audit['manifest_sha256'] != frozen['manifest_sha256']
            or summary['samples'] != frozen['split_counts']['smoke']):
        raise ValueError('Missing or inconsistent smoke audit')
    estimate = admission_seconds(summary['elapsed_seconds'], summary['samples'], frozen['split_counts']['full'])
    allowed = estimate <= cfg['full_gpu_seconds']
    write_new(folder/'admission.json', {
        'status': 'admitted' if allowed else 'blocked_time_budget',
        'source_commit': entry['code_commit'],
        'inference_source_commit': inference['code_commit'],
        'summary_sha256': sha(folder/'summary.json'), 'audit_sha256': sha(folder/'audit.json'),
        'estimated_full_seconds': estimate, 'full_limit_seconds': cfg['full_gpu_seconds'],
        'smoke_seconds': summary['elapsed_seconds'], 'smoke_samples': summary['samples'],
        'full_samples': frozen['split_counts']['full'], 'accuracy_used_for_admission': False,
        'gpu_used': False,
    })
    print('FSCD_LVIS_FULL_ADMITTED' if allowed else 'FSCD_LVIS_FULL_TIME_BUDGET_BLOCKED', flush=True)
    return 0 if allowed else 78


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(admit(args.root.resolve()))
