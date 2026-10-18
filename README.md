# Plan B：尺度穩健的示例引導物件計數

我們以 GeCo2 為基礎，研究尺度補充提示、切片推論與可訓練密度分支。輸入是一張 RGB 圖像及三個目標示例框；輸出是數量、密度圖和物件位置。

已完成 GeCo2／SAM2 的 1,286 張完整 val 基線評估及獨立核算：MAE 9.39、RMSE 33.26。資料共 82,042 個物件，每圖中位數 27、平均 63.80 個；至少 300 個物件的 39 張圖仍有 MAE 113.44 的明顯誤差。這是官方權重的復現，尚未重新訓練或取得新增方法的精度提升。原有 LOCA 實驗不等同於本路線的結果。

- [B0 實驗結果與數量背景](docs/B0_RESULTS.md)
- [M2／M3 固定實驗規則](docs/M2_M3_PROTOCOL.md)
- [M2／M3 第一輪實測結果](docs/M2_M3_RESULTS.md)
- [實施方案與階段門檻](docs/PLAN_B.md)
- [開發與實驗進度](docs/PROGRESS.md)
- [公開資料與隱私規則](docs/PUBLISHING.md)
- [環境建置腳本](scripts/bootstrap.sh)
- [B0／M1 執行與結果判讀](docs/REPRODUCTION.md)

執行環境目標：單張 48GB Ada GPU、Linux、Python 3.10、PyTorch 2.7.1。上游 GeCo2 固定於 `67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b`。模型及資料集另外下載，不放進 Git。

此倉庫只發布去識別化的程式、方法、設定、測試與結果。SSH 金鑰、帳密、服務端點、個人路徑及完整原始日誌均不發布。
