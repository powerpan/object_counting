"""Annotation/prediction diagram only; does not redistribute dataset images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def lookup(path, name):
    return next(row for row in map(json.loads, path.read_text().splitlines()) if row['image'] == name)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--image', default='3481.jpg'); args = p.parse_args(); root = args.root
    ann = json.loads((root / 'data/fsc147/annotations/annotation_FSC147_384.json').read_text())[args.image]
    base = lookup(root / 'results/b0-val-b1-boundary-r1/predictions.jsonl', args.image)
    m2 = lookup(root / 'results/m2-m3-v1/val-m2/predictions.jsonl', args.image)
    points = np.asarray(ann['points'])
    original = np.asarray(ann['box_examples_coordinates'])[:3, [0, 2]].reshape(-1, 4)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax in axes.flat:
        ax.scatter(points[:, 0], points[:, 1], s=5, color='#b7c2cb', label='Ground-truth point', zorder=1)
        ax.set(xlim=(0, base['width']), ylim=(base['height'], 0), aspect='equal')
        ax.set_xlabel('x / original-image pixels'); ax.set_ylabel('y / original-image pixels')
    for x0, y0, x1, y1 in original:
        axes[0, 0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#176baf', linewidth=1.8))
    axes[0, 0].set_title(f"Original 3 exemplars | ground truth: {len(points)}")
    for i, candidate in enumerate(m2['detail']['pseudo_exemplars']):
        x0, y0, x1, y1 = candidate['box_xyxy']
        axes[0, 1].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#d84b3b', linewidth=1.8))
        axes[0, 1].annotate(f"{i+1}: {candidate['annotated_points_inside']} targets", (x1, y0), xytext=(8, 2),
                            textcoords='offset points', fontsize=8, color='#a13227')
    axes[0, 1].set_title('M2 accepted pseudo exemplars (red)')
    for ax, row, title in [(axes[1, 0], base, 'B0'), (axes[1, 1], m2, 'M2')]:
        boxes = np.asarray(row['boxes_xyxy']).reshape(-1, 4)
        centers = (boxes[:, :2]+boxes[:, 2:])/2
        ax.scatter(centers[:, 0], centers[:, 1], marker='+', s=18, linewidth=.8, color='#125a49', label='Predicted centre')
        ax.set_title(f"{title}: {row['prediction']} | absolute error: {abs(row['prediction']-row['truth'])}")
        ax.legend(loc='lower right', fontsize=7)
    fig.suptitle(f"M2 diagnostic: {args.image} | points and boxes only; no source image", fontsize=12)
    out = root / 'results/m2-m3-v1/figures'; out.mkdir(exist_ok=True)
    fig.savefig(out / ('m2-case-'+Path(args.image).stem+'.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
