
# 🌏 台灣地震預測小專題（以 2025/7/5 為目標）

本專案透過台灣中央氣象局的歷史地震資料，建立兩種輕量級模型（Logistic Regression 與 MLP），預測 2025 年 7 月 5 日是否可能發生規模 ≥ 4 的地震。此專題以學術與實作為主，**不作為實際預警工具使用**。

---

## 📁 專案架構

```
earthquake_prediction/
│
├── data/                       # 原始與處理過的地震資料
│   ├── GDMScatalog.csv         # 原始下載資料（CSV）
│   └── processed/              # 處理後可用的特徵檔（features, labels）
│
├── notebooks/                  # Jupyter 分析筆記本
│   ├── main.py
│   └── prepare_dataset.py
│
├── src/                        # Python 模組化程式碼
│   ├── data_loader.py          # 資料處理模組
│   ├── models/                 # 模型定義
│   ├── train.py                # 訓練兩個模型來得到參數
│   ├── predict_mlp.py          # 使用 MLP 進行預測
│   └── predict_logreg.py       # 使用邏輯回歸進行預測
│
├── requirements.txt            # 套件需求列表
└── README.md                   # 專案說明文件
```

---

## 📊 訓練資料來源與範圍

| 項目        | 範圍或描述     |
|-------------|----------------|
| 起始日期     | 2000-01-01     |
| 結束日期     | 2025-05-25     |
| 經度範圍     | 118 ~ 126      |
| 緯度範圍     | 20 ~ 26        |
| 規模 ML 範圍 | 3 ~ 10         |
| 深度範圍     | 0 ~ 50 公里     |

---

## 🧩 特徵簡介（Features）

| 欄位名稱   | 說明                                   |
|------------|----------------------------------------|
| `lat`, `lon` | 經緯度位置（可計算地震分布密度）         |
| `depth`    | 地震深度（km）                          |
| `ML`       | 芮氏規模                                 |
| `nstn`     | 偵測站數量（數值越高代表可靠度越高）       |
| `dmin`     | 最近偵測站距離（km）                     |
| `gap`      | 方位角間隙，數值越高代表偵測角度越不完整     |
| `trms`     | 到時殘差的 RMS 值（秒）                   |
| `ERH`, `ERZ` | 水平與垂直誤差（km）                     |
| `nph`      | 到時資料筆數                             |
| `quality`  | 品質等級（A~E）                         |
| `date`     | 可轉為週期性時間特徵（例如月、週等）         |

---

## 🧠 模型比較

| 項目        | Logistic Regression             | MLP (Neural Network)          |
|-------------|----------------------------------|-------------------------------|
| 模型複雜度   | 低（單層線性分類器）               | 中（多層非線性）              |
| 可解釋性     | 高，可直接看權重                    | 低，屬於黑盒模型              |
| 訓練速度     | 非常快                            | 中等，需 GPU 效率較佳         |
| 表現        | 準確率 99.26%，AUC 0.9998        | 準確率 98.47%，AUC 0.9991     |
| 適用場景     | 特徵維度少、關係較線性               | 特徵多、關係複雜、需強表達能力時 |
| 過擬合風險   | 低                                | 中等，需搭配 Early Stopping  |

---

## 📌 推論設定

- **輸入資料時間範圍**：2025/3/1 ～ 2025/4/29
- **預測目標**：2025/7/5 是否可能發生 ML ≥ 4 的地震
- **推論模型**：MLP 或 Logistic Regression（二擇一）
- **預測輸出**：
  - 機率值（0~1）
  - 分類結果（是否可能發生地震）

---

## 🔍 範例推論結果

### ✅ 使用 MLP 預測
```
模型預測 2025/07/05 發生 ML≥4 地震的機率為：0.3436
預測分類結果：不太可能
```

### ✅ 使用 Logistic Regression 預測
```
➡ 使用最新一筆資料預測 2025/07/05 是否可能地震
➡ 發生 ML ≥ 4 地震的機率為：1.0000
➡ 是否可能發生地震：是
```

> ⚠ 注意：上述推論結果純屬學術性模擬，實際地震預測極為困難，機率值不代表真實發生機率。

---

## 🚀 使用方式

1. 開啟終端機，切換目錄：  
   ```bash
   cd src
   ```

2. 修改路徑（predict 檔案中需手動設定資料路徑為你的絕對路徑）。

3. 執行預測：
   ```bash
   python predict_mlp.py
   # 或
   python predict_logreg.py
   ```

4. 查看預測結果（會印出地震機率與分類判斷）。

---

## 📦 環境需求

請安裝以下 Python 套件：
```
pandas
numpy
torch
scikit-learn
matplotlib
```
可用 `pip install -r requirements.txt` 快速安裝。

---


"# predict-earthquake" 
