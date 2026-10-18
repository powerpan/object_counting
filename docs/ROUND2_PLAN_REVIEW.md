# 第二輪方案與現有程式的核對

日期：2026-10-18。此檢查只讀固定版原始碼、既有結果及資料清單；尚未執行第二輪 GPU gate、容量試驗、推論或訓練。

| 草案中的問題 | 開跑前固定的處理 | 依據 |
|---|---|---|
| 最後上採樣層未具名 | `adapt_features.up3`，卷積 `adapt_features.up3.layer.1`；保留其餘凍結 | `models/query_generator.py` 的 up1→up2→up3 與 `models/regression_head.py` |
| 輸入特徵尺寸未驗證 | 靜態程式為 256×512×512，再串接 1 通道原始 centerness；GPU gate 仍須確認 | 64→128→256→512 上採樣及 `class_embed` |
| 前置尺度抖動被抵消 | 基準縮放後乘 U[0.8,1.0]；同步圖、框、點，保留畫布與示例限制 | `utils.data.resize_and_pad` 依長邊及示例重新定尺度 |
| 單圖 adapter 與 batch 4／8 不合 | 凍結特徵逐圖即時計算，密度分支組批；batch 8 先做容量推估 | `QueryAdapter.appearance` 限制 batch=1 |
| Softplus 初始總量與 100 倍單位含糊 | 固定 Q=100D，初始化 Q=0.01／有效像素，padding=0 | 512² 網格直接積分會放大初始偏置 |
| 融合格計數與最終密度不相符 | 融合用 M1 格內質量，中心數只作特徵 | M1 高斯可能跨格，格內質量不等於中心數 |
| logistic regression 名稱與平方損失不符 | 明確採 sigmoid 線性權重回歸，固定標準化、正則化、優化器 | 計數最小平方目標不屬二元分類 likelihood |
| 同折選寬度、擬合 gate 的邊界不明 | 所有選擇、標準化、gate、組合納入都限擬合折 | 29 類分為 15／14 類，但圖片為 496／790 張 |
| 三種子與兩折寬度可能不同 | seed42 的正式結果維持折外；43／44只重訓全val選定寬度，明列條件式開發敏感度 | 防止把全val選寬度後的重訓冒充獨立選擇驗證 |
| M4-ft 需完整框監督 | 先核驗官方 FSCD 真實框及原損失；不可用點合成框替代，不可用 `counter_infer` 偽裝完整訓練入口 | `utils.data.FSC147DATASET` 需要 `instances_train.json`；`train.py` 同時算主／輔助損失 |
| 密度輸出沒有離散定位規則 | F1只報檢測中心，密度／融合以 GAME 評估空間分布 | 不另加未登記的峰值閾值 |
| 3×3 不一定放大2.5倍 | 依裁片範圍及示例限制記錄真實倍率 | halo固定為全圖10%，邊角與中間裁片寬度不同 |

已重算 B0 有方向平均誤差為 −4.24495，與草案 −4.24 相符。train-holdout 固定八類為 bottles、caps、coins、fishes、kidney beans、naan bread、peppers、spring rolls，共 202 張；擬合資料為 81 類、3,457 張。完整清單與雜湊另存設定檔。

正式規則見 [ROUND2_PROTOCOL.md](ROUND2_PROTOCOL.md)。原草案 SHA-256：`44f2ec347cd46c9f9d8c6f8ba64a6b4660e797f71dd379996301af2c8ac02333`。所有更動在第一次第二輪運算之前提交；原草案檔未修改。尚待真實 GPU 等價、梯度、容量、資料及凍結參數檢查，不能把這份靜態核對當成已通過實作驗證。
