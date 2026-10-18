"""Export paired uncertainty plots from the five audited validation runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); result = args.root / 'results/m2-m3-v1'
    names = ['m2', 'm2-repeat', 'm3', 'm3-repeat', 'm2-m3']
    labels = ['M2', 'M2 repeated exemplars', 'M3', 'M3 repeated full image', 'M2 + M3']
    reports = [json.loads((result / ('val-'+name) / 'analysis.json').read_text()) for name in names]
    assert all(r['status'] == 'passed' and r['n'] == 1286 for r in reports)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (group, metric, title) in zip(axes.flat, [
            ('overall', 'mae', 'MAE change | all 1,286 images'),
            ('overall', 'rmse', 'RMSE change | all 1,286 images'),
            ('>=300', 'mae', 'MAE change | 39 dense images'),
            ('>=300', 'rmse', 'RMSE change | 39 dense images')]):
        for i, report in enumerate(reports):
            for unit, color, offset in [('image', '#176baf', -.13), ('class', '#c8722a', .13)]:
                d = report['paired_vs_b0'][group][unit]
                lo, hi = d[metric+'_delta_95ci']; point = d[metric+'_delta']
                ax.plot([lo, hi], [i+offset]*2, color=color, linewidth=2)
                ax.scatter([point], [i+offset], s=24, color=color, zorder=3)
        ax.axvline(0, color='#555', linewidth=.8)
        ax.set_yticks(range(5), labels); ax.invert_yaxis(); ax.set_title(title, fontsize=11)
        ax.set_xlabel('Change from B0 (objects): lower is better')
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Paired bootstrap 95% intervals | validation only, 2,000 resamples', fontsize=13)
    fig.legend(handles=[Line2D([0], [0], color='#176baf', marker='o', label='Image resampling'),
                        Line2D([0], [0], color='#c8722a', marker='o', label='Class resampling')],
               loc='outside lower center', ncol=2, frameon=False)
    out = result / 'figures'; out.mkdir(exist_ok=True)
    fig.savefig(out / 'paired-intervals.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
