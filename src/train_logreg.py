# src/train_logreg.py

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 設定路徑
features_path = r'D:/4_AI/arthquake_prediction/arthquake_prediction/data/processed/train_features.csv'
labels_path = r'C:/Users/eejia/Desktop/arthquake_prediction/data/processed/train_labels.csv'
model_save_path = r'C:/Users/eejia/Desktop/arthquake_prediction/models/logreg_model.pkl'

# 載入資料
X = pd.read_csv(features_path).values
y = pd.read_csv(labels_path).values.ravel()

# 分割資料集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("準確率 (Accuracy):", accuracy_score(y_val, y_pred))
print("精確率 (Precision):", precision_score(y_val, y_pred))
print("召回率 (Recall):", recall_score(y_val, y_pred))
print("F1 分數:", f1_score(y_val, y_pred))
print("ROC AUC 分數:", roc_auc_score(y_val, y_prob))

# 儲存模型
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(model, model_save_path)
print(f"模型已儲存至 {model_save_path}")
