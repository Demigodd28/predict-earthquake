import pandas as pd
from datetime import datetime
import os

def load_earthquake_data():
    """
    讀取 data/GDMScatalog.csv 檔案，並處理日期時間
    """
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'GDMScatalog.csv')
    df = pd.read_csv(filepath)

    # 合併日期與時間成 datetime 欄位
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df['date_only'] = df['datetime'].dt.date

    return df


def extract_daily_features(df):
    """
    將地震事件依日期彙總，產生每日特徵與 label(是否有 ML >= 3 的地震)
    """
    # 彙整每日的統計特徵
    daily_stats = df.groupby('date_only').agg({
        'ML': ['mean', 'max', 'count'],
        'depth': ['mean', 'max'],
        'lat': 'mean',
        'lon': 'mean'
    })

    # 攤平成單層欄位名稱
    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
    daily_stats.reset_index(inplace=True)

    # 加入 label（該日是否有 ML ≥ 4）
    label_series = df[df['ML'] >= 4].groupby('date_only').size()
    daily_stats['label'] = daily_stats['date_only'].map(label_series).fillna(0).astype(int)
    daily_stats['label'] = daily_stats['label'].apply(lambda x: 1 if x > 0 else 0)


    return daily_stats