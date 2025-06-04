# src/predict_logreg.py

import pandas as pd
import joblib
import os

# === 路徑設定 ===
MODEL_PATH = r"C:/Users/eejia/Desktop/arthquake_prediction/models/logreg_model.pkl"
INFER_PATH = r"C:/Users/eejia/Desktop/arthquake_prediction/data/processed/inference_input.csv"

# === 檢查模型是否存在 ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型檔案不存在：{MODEL_PATH}")

# === 載入模型 ===
model = joblib.load(MODEL_PATH)

# === 載入推論資料 ===
inference_data = pd.read_csv(INFER_PATH).values

# === 使用最新一筆資料進行推論 ===
latest_input = inference_data[-1].reshape(1, -1)

prob = model.predict_proba(latest_input)[0, 1]
pred = model.predict(latest_input)[0]

# === 輸出結果 ===
print("\n🔍 預測結果（使用 Logistic Regression）:")
print(f"➡ 使用最新一筆資料預測 2025/07/05 是否可能地震")
print(f"➡ 發生 ML ≥ 4 地震的機率為：{prob:.4f}")
print(f"➡ 是否可能發生地震：{'是' if pred == 1 else '否'}")
