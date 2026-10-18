# Plan B：尺度穩健的示例引導物件計數

我們以 GeCo2 為基礎，研究尺度補充提示、切片推論與可訓練密度分支。輸入是一張 RGB 圖像及三個目標示例框；輸出是數量、密度圖和物件位置。

目前進入環境建置與官方基線復現階段。尚未完成 GeCo2 全驗證集評估，也沒有新的精度提升結論。原有 LOCA 實驗不等同於本路線的結果。

- [實施方案與階段門檻](docs/PLAN_B.md)
- [開發與實驗進度](docs/PROGRESS.md)
- [公開資料與隱私規則](docs/PUBLISHING.md)
- [環境建置腳本](scripts/bootstrap.sh)
- [B0／M1 執行與結果判讀](docs/REPRODUCTION.md)

執行環境目標：單張 48GB Ada GPU、Linux、Python 3.10、PyTorch 2.7.1。上游 GeCo2 固定於 `67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b`。模型及資料集另外下載，不放進 Git。

此倉庫只發布去識別化的程式、方法、設定、測試與結果。SSH 金鑰、帳密、服務端點、個人路徑及完整原始日誌均不發布。
