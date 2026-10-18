# Plan B：尺度穩健的示例引導物件計數

我們以 GeCo2 為基礎，研究尺度補充提示、切片推論與可訓練密度分支。輸入是一張 RGB 圖像及三個目標示例框；輸出是數量、密度圖和物件位置。

已完成 GeCo2／SAM2 的 B0 基線及 M2、M3 與對照共五組完整 val 評估，每組均為 1,286 張並通過獨立核算。資料共 82,042 個物件，每圖中位數 27、平均 63.80 個，範圍 7–2,092。

目前保留 B0：MAE 9.39、RMSE 33.26。M2 為 10.32／37.03；M3 為 9.09／34.41，平均誤差下降但 RMSE 上升，且差值區間跨零；M2＋M3 為 9.70／37.28。五組皆未通過事前候選門檻。M2 出現多物件候選示例，M3 的兩張極端案例則顯示切片也改變上游 SAM 遮罩執行分支；這些是後續研究線索，不是已證明的改進。完整負結果、對照、區間與案例均公開。

第二輪 M4-head-S 已完成 30 輪及全部 1,286 張 val，獨立核算通過。S 單獨計數的 MAE／RMSE 為 **16.62／83.30**，密集組 MAE 為 **300.83**，明顯差於 B0；39 張密集圖片全部少計。原 B0 檢測框逐位元不變，保留 B0。L 依明確指示於第 2 輪取消，沒有完整驗證結果，不能判斷其最終效果。寬度選擇及原 S 種子重訓取消；最終 test 及外部確認尚未啟動。原有 LOCA 實驗不等同於本路線的結果。

M4-v2 已先登記 M1 殘差設計、固定損失與取樣、396 張留出集和第 10 輪停止規則，再開始實測。S 訓練集診斷發現，原擬合集 15 張 ≥500 圖的預測／真值中位數只有 **35.4%**；這是擬合問題線索，尚未證明個別原因。S 的兩折區域融合已完成並核算，依事前修訂只作紀錄。v2 的起點等價、batch 8 容量及 B0 同集留出參照均通過，但 seed 42 第 10 輪的留出 RMSE 為 **17.08497**，高於 B0 的 **16.52019**，已於 2026-10-18 11:47 UTC 按規則自動停止，獨立核算通過。留出 MAE 雖下降約 2.10%，RMSE 卻上升約 3.42%；未完成 30 輪、未執行 v2 正式 val 或 test，沒有自動重訓或追加種子。

- [B0 實驗結果與數量背景](docs/B0_RESULTS.md)
- [M2／M3 固定實驗規則](docs/M2_M3_PROTOCOL.md)
- [M2／M3 第一輪實測結果](docs/M2_M3_RESULTS.md)
- [第二輪開跑前審查](docs/ROUND2_PLAN_REVIEW.md)
- [第二輪固定規則](docs/ROUND2_PROTOCOL.md)
- [第二輪啟動與容量實測](docs/ROUND2_PREFLIGHT_RESULTS.md)
- [L 型時間預算修訂](docs/ROUND2_BUDGET_AMENDMENT.md)
- [M4 訓練後驗證與合計預算](docs/ROUND2_VALIDATION_WORKFLOW.md)
- [S 型先驗證的順序修訂](docs/ROUND2_S_FIRST_AMENDMENT.md)
- [S 型訓練與驗證結果](docs/M4_S_RESULTS.md)
- [L 型取消與保存記錄](docs/M4_L_CANCELLATION.md)
- [取消 L 的實驗修訂與時間線](docs/ROUND2_L_CANCEL_AMENDMENT.md)
- [M4-v2 固定規則](docs/M4_V2_PROTOCOL.md)
- [M4-v2 實作核對](docs/M4_V2_IMPLEMENTATION_REVIEW.md)
- [M4-v2 前置結果與 S 診斷](docs/M4_V2_START_RESULTS.md)
- [M4-v2 第 10 輪停止結果](docs/M4_V2_STOP_RESULTS.md)
- [實施方案與階段門檻](docs/PLAN_B.md)
- [開發與實驗進度](docs/PROGRESS.md)
- [公開資料與隱私規則](docs/PUBLISHING.md)
- [環境建置腳本](scripts/bootstrap.sh)
- [實驗重現與結果核算](docs/REPRODUCTION.md)

執行環境目標：單張 48GB Ada GPU、Linux、Python 3.10、PyTorch 2.7.1。上游 GeCo2 固定於 `67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b`。模型及資料集另外下載，不放進 Git。

此倉庫只發布去識別化的程式、方法、設定、測試與結果。SSH 金鑰、帳密、服務端點、個人路徑及完整原始日誌均不發布。
