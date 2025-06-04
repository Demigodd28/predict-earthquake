import pandas as pd
from datetime import datetime

def prepare_dataset(input_csv, output_dir):
    # 讀取原始資料
    df = pd.read_csv(input_csv)

    # 建立 datetime 欄位
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['date_only'] = df['datetime'].dt.date

    # 製作每日統計特徵
    daily_stats = df.groupby('date_only').agg({
        'ML': ['mean', 'max', 'count'],
        'depth': ['mean', 'max'],
        'lat': 'mean',
        'lon': 'mean'
    })
    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
    daily_stats.reset_index(inplace=True)

    # 製作 label（是否有 ML >= 4）
    label_series = df[df['ML'] >= 4].groupby('date_only').size()
    daily_stats['label'] = daily_stats['date_only'].map(label_series).fillna(0).astype(int)
    daily_stats['label'] = daily_stats['label'].apply(lambda x: 1 if x > 0 else 0)

    # 分離訓練集與推論輸入（2025年3~4月）
    daily_stats['date_only'] = pd.to_datetime(daily_stats['date_only'])
    train_data = daily_stats[daily_stats['date_only'].dt.year < 2025]
    inference_data = daily_stats[
        (daily_stats['date_only'] >= '2025-03-01') &
        (daily_stats['date_only'] <= '2025-04-30')
    ]

    # 儲存訓練特徵與標籤
    train_features = train_data.drop(columns=['date_only', 'label'])
    train_labels = train_data['label']
    train_features.to_csv(f'{output_dir}/train_features.csv', index=False)
    train_labels.to_csv(f'{output_dir}/train_labels.csv', index=False)

    # 儲存推論輸入資料
    inference_features = inference_data.drop(columns=['date_only', 'label'])
    inference_features.to_csv(f'{output_dir}/inference_input.csv', index=False)

    print("✅ 資料已成功處理與儲存")

# 範例用法（請依實際路徑修改）
# prepare_dataset('data/GDMScatalog.csv', 'data/processed')
