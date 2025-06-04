import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
from models.MLP import EarthquakePredictor

# 路徑設定
features_path = r'C:/Users/eejia/Desktop/arthquake_prediction/data/processed/train_features.csv'
labels_path = r'C:/Users/eejia/Desktop/arthquake_prediction/data/processed/train_labels.csv'
model_save_path = r'C:/Users/eejia/Desktop/arthquake_prediction/models/best_model.pth'

# 載入資料
X = pd.read_csv(features_path).values
y = pd.read_csv(labels_path).values.ravel()

# 分割訓練與驗證
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tensor dataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 模型、loss、optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EarthquakePredictor(input_dim=X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 確保儲存模型的資料夾存在
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Early stopping & checkpoint
best_val_loss = float('inf')
patience = 5
wait = 0

for epoch in range(50):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 驗證
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        wait += 1
        if wait >= patience:
            print("早停 triggered.")
            break
