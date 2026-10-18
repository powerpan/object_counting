# 上游程式來源

`planb/geco_adapter.py` 的特徵擷取與 query 解碼流程改寫自 [GeCo2](https://github.com/jerpelhan/GECO2) 的 `models/counter_infer.py`，固定版本為 `67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b`。原程式採 MIT License；完整版權與授權文字保留於 [GECO2_LICENSE.txt](third_party/GECO2_LICENSE.txt)。

本倉庫的改動將主幹、示例特徵與解碼步驟分開，以支援同圖特徵重用及裁片外示例；沒有改寫或重新分發 SAM2、Deformable-DETR 的原始碼和權重。這些上游依賴仍適用各自的授權，下載後保留在執行環境。
