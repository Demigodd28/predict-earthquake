import torch
import pandas as pd
import numpy as np
import os
from models.MLP import EarthquakePredictor

# ========= 設定路徑 =========
model_path = r'C:/Users/eejia/Desktop/arthquake_prediction/models/best_model.pth'
input_path = r'C:/Users/eejia/Desktop/arthquake_prediction/data/processed/inference_input.csv'

# ========= 載入模型 =========
inference_data = pd.read_csv(input_path).values
input_dim = inference_data.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EarthquakePredictor(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========= 將多筆特徵整合為單一輸入（平均） =========
# 這是重點：將 3~4 月的多筆紀錄平均成一筆代表性輸入
mean_input = np.mean(inference_data, axis=0, keepdims=True)

with torch.no_grad():
    input_tensor = torch.tensor(mean_input, dtype=torch.float32).to(device)
    pred = model(input_tensor)
    prob = pred.item()  # 單一機率值
    label = int(prob >= 0.5)

# ========= 顯示結果 =========
print(f"模型預測 2025/07/05 發生 ML≥4 地震的機率為：{prob:.4f}")
print(f"預測分類結果：{'可能發生' if label == 1 else '不太可能'}")
