"""Write a concise Traditional Chinese report from independently audited runs."""
import argparse
import json
from pathlib import Path


LABELS = {'m2': 'M2 新尺度示例', 'm2-repeat': 'M2 重複示例對照', 'm3': 'M3 重疊切片',
          'm3-repeat': 'M3 重複全圖對照', 'm2-m3': 'M2＋M3'}


def interval(x):
    return f'{x[0]:+.2f} 至 {x[1]:+.2f}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    args = p.parse_args(); repo = args.repo
    baseline = json.loads((repo / 'results/b0-val-b1-boundary-r1/summary.json').read_text())
    root = repo / 'results/m2-m3-v1'
    completed = {}
    for method in LABELS:
        d = root / ('val-' + method)
        if not (d / 'analysis.json').exists(): continue
        a = json.loads((d / 'analysis.json').read_text())
        s = json.loads((d / 'summary.json').read_text())
        assert a['status'] == 'passed' and s['status'] == 'complete'
        completed[method] = (s, a)
    lines = ['# M2／M3 第一輪實驗結果', '',
             '日期：2026-10-18。設定：`m2-m3-v1`。所有數值由實際逐圖預測產生，並通過獨立核算。', '',
             f'目前已完成並核算 {len(completed)}／5 組。' + ('完整結果如下。' if len(completed) == 5 else '未列出的組別仍待完成，不能由目前結果推定其表現。'), '',
             '我們沿用凍結的 GeCo2／SAM2 官方權重，在 FSC-147 官方 val 的全部 1,286 張圖片比較。真值共 82,042 個物件，每圖中位數 27、平均 63.80 個，範圍 7–2,092。MAE 是每張多數／少數的平均絕對數量，RMSE 對嚴重失誤較敏感；兩者不是百分比。這輪沒有訓練，也沒有執行 test。', '',
             '## 整體與密集圖片', '',
             '| 方法 | MAE | RMSE | ≥300 個組 MAE（39 圖） | GAME(1)／GAME(2) | 主循環秒數 | allocated 峰值 MiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def row(label, s):
        m = s['overall']
        return f"| {label} | {m['mae']:.3f} | {m['rmse']:.3f} | {s['groups']['>=300']['mae']:.3f} | {m['game1']:.2f}／{m['game2']:.2f} | {s['elapsed_seconds']:.1f} | {s['peak_allocated_mib']:.0f} |"
    lines.append(row('B0／M1 基準', baseline))
    for method, (s, a) in completed.items(): lines.append(row(LABELS[method], s))
    lines += ['', 'GAME 是將圖片分成 2×2 或 4×4 網格後，各區計數誤差的總和再對圖片平均。密度圖由檢測框生成並守恆，未訓練密度分支。時間是單次作業的完整主循環，不含環境安裝和模型初始化；各組流程開銷不同，不能把它當成嚴格 FLOPs 比較。', '',
              '## 配對差值與判讀', '',
              '以下差值均為「新方法 − B0」，負值較好。95% 區間各重抽樣 2,000 次；圖片配對與類別配對分開列出。類別重抽樣保留該類別全部圖片的權重。這是開發用 val，不是獨立確認，也沒有校正多組比較。', '',
              '| 方法 | MAE 差值 | 圖片配對 95% 區間 | 類別配對 95% 區間 | RMSE 差值 | 通過本輪候選門檻 |',
              '|---|---:|---|---|---:|---|']
    for method, (s, a) in completed.items():
        ci = a['paired_vs_b0']['overall']; img, cls = ci['image'], ci['class']
        lines.append(f"| {LABELS[method]} | {img['mae_delta']:+.3f} | {interval(img['mae_delta_95ci'])} | {interval(cls['mae_delta_95ci'])} | {img['rmse_delta']:+.3f} | {'是' if a['val_candidate_gate'] else '否'} |")
    lines += ['', '候選門檻為：整體 MAE 不超過 B0 的 102%，同時 RMSE 和密集組 MAE 都下降；還須看對照及區間，不能僅憑此表宣布穩定改善。', '',
              '## 觸發與失敗線索', '',
              '| 方法 | M2 觸發圖數 | M3 觸發圖數 | 改善／退步／誤差不變圖數 |',
              '|---|---:|---:|---:|']
    for method, (s, a) in completed.items():
        lines.append(f"| {LABELS[method]} | {s['m2_triggered_images']} | {s['m3_triggered_images']} | {a['improved_images']}／{a['worsened_images']}／{a['unchanged_error_images']} |")
    for method in ('m2', 'm2-m3'):
        if method in completed:
            a = completed[method][1]
            lines += ['', f"{LABELS[method]} 共選出 {a['pseudo_candidates']} 個候選示例，其中 {a['pseudo_candidates_with_annotation_inside']} 個框含至少一個官方目標點，{a['pseudo_candidates_with_multiple_annotations_inside']} 個含多個點。這是候選品質的代理檢查，不等同人工確認語意正確；多物件框可能混淆示例，不能直接當成可靠單一物件。"]
    if (root / 'figures/m2-case-3481.png').exists():
        lines += ['', '以 `3481.jpg` 為例，真值 431 個，B0 預測 316，M2 預測 29。一個通過篩選的增補框包住了 67 個標註點；高外觀相似度仍可能選到整段圖案。這是事後診斷，沒有據此更動本輪參數。下圖只畫標註點、示例框及檢測中心，沒有使用資料集原圖。', '',
                  '![M2 多物件示例失敗案例](../results/m2-m3-v1/figures/m2-case-3481.png)']
    lines += ['', '各組最大改善／退步案例及類別、完整密度分組指標與區間保存在對應 `analysis.json`／`summary.json`。推論策略的輸入沒有真值；目標點只在推論結束後作計分與候選品質檢查。', '',
              '## 可核對證據', '',
              '原模型與新介面在涵蓋三種密度的 32 張圖上，原始框、objectness、SAM 分數逐位元相同；最終框與保存的 B0 相同。每一個完整實驗又逐圖核對第一輪框與 B0 完全一致。密度／幾何及統計共 11 項 CPU 測試通過；完整輸出再次對照原始真值、指標和檔案雜湊。', '']
    for method in completed:
        path = '../results/m2-m3-v1/val-' + method
        lines.append(f'- {LABELS[method]}：[摘要]({path}/summary.json)、[獨立核算與配對區間]({path}/analysis.json)、[逐圖預測]({path}/predictions.jsonl)、[執行設定]({path}/protocol.json)')
    contrast_file = root / 'contrasts.json'
    if contrast_file.exists():
        contrasts = json.loads(contrast_file.read_text())['contrasts']
        if contrasts:
            lines += ['', '## 與對照組的直接比較', '',
                      '| 比較（前者 − 後者） | MAE 差值 | 圖片配對 95% 區間 | 類別配對 95% 區間 |',
                      '|---|---:|---|---|']
            for key, values in contrasts.items():
                left, right = key.split('_vs_')
                lines.append(f"| {LABELS[left]} − {LABELS[right]} | {values['image']['mae_delta']:+.3f} | {interval(values['image']['mae_delta_95ci'])} | {interval(values['class']['mae_delta_95ci'])} |")
            lines += ['', '對照組控制示例數或推論次數，仍有特徵和遮罩候選數差異；完整配對資料見 [contrasts.json](../results/m2-m3-v1/contrasts.json)。']
    lines += ['', '方法、事前規則與限制見 [M2／M3 規則](M2_M3_PROTOCOL.md)，原基準見 [B0 結果](B0_RESULTS.md)。未以不佳結果為由修改閾值，也未選取部分圖片作為正式結果。']
    (repo / 'docs/M2_M3_RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
