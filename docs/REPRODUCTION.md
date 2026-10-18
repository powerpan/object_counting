# B0／M1 執行與判讀

先執行 `scripts/bootstrap.sh` 建置環境。一般 Python 依賴預設使用 HTTPS 套件鏡像，PyTorch／Torchvision 的 cu118 wheel 來自官方索引；可用 `PLANB_PIP_INDEX_URL` 更換一般依賴來源。CUDA 11.8 工具包必須已存在。

## 輸入檔案

`scripts/fetch_assets.py` 分別下載 FSC 專用 GeCo2 權重、SAM2 權重、FSC-147 官方圖像壓縮包和標註。執行時指定 `--root` 和 `--part`。Google Drive 下載支援中斷續傳，完成後保存來源、大小及 SHA-256。這些值能檢查搬移前後是否一致，不能當作作者提供的數位簽章。

本輪改為伺服器直接下載公開資源。GeCo2 預設下載作者 Hugging Face 倉庫的 FSC 專用權重，固定 revision `ed3c8ff3753e731fd7074862c0ea49d908785335`；保留 `--part geco-original` 供原 Google Drive 版本的核對使用。兩個檔案的容器大小不同，必須用 `scripts/check_checkpoint_equivalence.py` 對照 `manifests/geco_gdrive_tensors.json` 的 475 個模型張量，逐項確認形狀、資料型別和完整內容雜湊一致；檔案名稱相同不是等價證據。

FSC-147 的 val 圖像使用 `scripts/fetch_validation.py --root "$WORK_ROOT" --manifest "$CODE_ROOT/manifests/fsc147_val.json"`。傳輸來源是第三方 `isentropic/FSC147` 鏡像的固定版本 `3e420cb6537e803dd6d4516623ce82a79c0317b8`，每一張圖均對照原官方壓縮包中的檔案雜湊。三份標註由原始 GitHub 取得，也逐檔核對。只有全部 1,289 個檔案一致才寫入完成憑據。模型和資料不需要透過本機上傳；本機只傳送程式與小型核驗清單。

`scripts/download_verified_ranges.py` 可對支援 HTTP Range 的大型公開資源使用 8 個並行連線。每段核對 Content-Range 和長度，最後核對完整 SHA-256；只有完整通過才提升為正式檔案。中斷前的前綴檔案需先停止寫入，才能用於續傳。

工作根目錄應包含：

```text
external/GECO2/                         # 固定版本的上游原始碼
weights/GECO2FSCD.pth
cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt
data/fsc147/images_384_VarV2/
data/fsc147/annotations/annotation_FSC147_384.json
data/fsc147/annotations/Train_Test_Val_FSC_147.json
```

B0 只需要 val 圖像；可以從原始壓縮包依官方 val 清單選取，減少傳輸量。共用標註 JSON 包含多個 split，本入口只取 val 對應記錄，不以 test 記錄作評估或選參。

## 執行順序

以下 `WORK_ROOT` 為上述工作根目錄，`CODE_ROOT` 為本倉庫路徑；兩者可以相同。

```bash
source "$WORK_ROOT/.venv/bin/activate"
python "$CODE_ROOT/scripts/check_cuda_ops.py" \
  --upstream "$WORK_ROOT/external/GECO2" \
  --output "$WORK_ROOT/results/cuda_ops.json"
python "$CODE_ROOT/scripts/evaluate_val.py" \
  --root "$WORK_ROOT" --output "$WORK_ROOT/results/b0-smoke-b1" \
  --limit 20 --batch-size 1
```

先檢查 `checkpoint_load.json`、逐圖預測及 `summary.json`。算子數值檢查、權重載入和 smoke 均通過後，才以新的輸出目錄執行 `--limit 0` 的完整 val。batch 2、4、8 為另立目錄的容量測試，須先確認顯存與輸出一致性。已有逐圖結果的目錄會拒絕覆寫。

`scripts/run_b0_when_ready.py` 將上述步驟串接：先等候建置成功與三份資產雜湊通過，再解壓僅含 val 圖像的資料包、執行算子和 M1 測試、20 張 smoke，最後跑完整 val。它讀取 `provenance/geco.json`、`sam.json`、`validation_bundle.json`，每份包含相對 `path`、`bytes`、`sha256`。資料包必須只含 `data/fsc147/` 之下的一般檔案。

此流程使用單程序鎖避免重複執行，準備階段最多等待 90 分鐘，smoke 最多 15 分鐘，完整 val 最多 90 分鐘。任何步驟失敗就停止，狀態寫入 `results/b0_workflow.json`；不會自動重跑、跳過失敗檢查或啟動訓練。本輪基線統一 batch 1，較大批次另外量測。遠端工作可在 SSH 斷線後繼續，但原始資料搬移仍需確認來源程序存活。

修復後重跑可指定 `--run-label boundary-r1`，為日誌及 smoke／完整 val 輸出加上新後綴，保留先前證據。這個選項不跳過核驗，也不是無條件自動重試；重啟前須先釐清失敗原因。

若存在 `provenance/validation_manifest.json`，流程會改用直接下載模式：檢查 `validation_download.json` 的完成狀態、清單雜湊，再重核每份實際檔案，不等待或解壓舊資料包。AutoDL 的 GitHub 加速可用 `PLANB_GIT_ACCELERATION=1`，只作用於 Git 複製步驟，不套用至一般依賴下載。

## 評估口徑

輸入為 RGB 圖像及官方前三個示例框。沿用固定版上游的縮放、示例尺度限制、1024 補邊、ImageNet 正規化，以及 0.11 相對閾值／0.5 NMS。補邊區域的檢測會排除。這個入口沒有 test 選項。

真實數量採官方點標註的個數。上游入口以預先生成密度圖的總和作真值，可能存在浮點或生成差異，因此完整復現仍須核對評估口徑，不能直接把本入口的數值當成論文數值的逐位元重現。MAE 是每張圖平均少數／多數的絕對數量；RMSE 會更重罰少數誤差很大的圖。摘要同時列樣本數、真值最小／中位／平均／最大值及總物件數。

M1 為每個最終檢測框產生總量為 1 的高斯分布，邊界截斷後重新歸一化。它不改計數，也不能自行補回漏檢。GAME(1)、GAME(2) 分別將圖像切成 2×2、4×4 區域，再加總各區的絕對計數誤差；數值低代表這個分區尺度下的空間分布較一致，不等同逐物件定位正確。位於圖像最遠邊界上的真值點歸到最後一個像素範圍，保留點數。

實際核驗發現 `1989.jpg` 的寬度為 408，其中一個點的 x 座標為 `408.00000000000006`。原檢查把一個 float64 間距的尾差判作越界，導致首次完整評估在 342 張後停止。修復只容許各圖像尺寸處 8 個 float64 可表示間距內的尾差，再歸入邊緣網格；更大的越界及非有限值仍拒絕。原始標註檔不修改，也不刪點或更動真實數量。全 val 預檢只有這一點屬於尾差越界；含精確邊界在內，共 9 張圖、19 個點需要網格邊界處理。`protocol.json` 記錄這項規則與評估程式雜湊，各圖結果記錄處理點數。

FSC 權重包含訓練輔助頭；推論不使用它們。SAM 遮罩修正模組則由另外的官方 SAM2 權重初始化。本入口會逐項記錄缺失與多餘鍵，只允許這兩類已知差異；其他不匹配立即停止。

20 張 smoke 採官方 val 清單前 20 張，只確認流程、數值和資源需求；不能代表整體精度。每圖時間包含模型內遮罩修正與同步，但不包含圖像讀取、密度圖生成及結果寫入。完整流程時間另列。峰值顯存同時報告 allocated 和 reserved。

完整評估完成後執行獨立核算，不直接把執行日誌視為驗收：

```bash
python "$CODE_ROOT/scripts/audit_b0_results.py" \
  --root "$WORK_ROOT" \
  --result "$WORK_ROOT/results/b0-val-b1-boundary-r1" \
  --previous-result "$WORK_ROOT/results/b0-val-b1"
```

這個核算程式不匯入推論入口，從逐圖結果及原始點標註重算整體和分組指標，檢查 val 清單、程式及標註雜湊、密度總量和有限值，並比對修復前已有圖片的檢測框及計數。`--previous-result` 為可選；本次用於證明邊界修復沒有改變前 342 張的模型輸出。成功才產生 `audit.json`。

目前 M1 的 CPU 單元測試可以在具備 NumPy 的環境執行：

```bash
python -m unittest discover -s tests -v
```

實際完成到哪一步，以 [進度記錄](PROGRESS.md) 為準。
