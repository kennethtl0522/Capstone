import pandas as pd

# 加载股票数据
stock_data = pd.read_csv('./stock/features.csv', parse_dates=['Date'])
# 加载情感数据
sentiment_data = pd.read_csv('./stock/classifier_results.csv', parse_dates=['datetime'])


stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None).dt.date
sentiment_data['datetime'] = pd.to_datetime(sentiment_data['datetime']).dt.date

# 聚合情感数据（按日期计算平均分数，取主要情感标签）
sentiment_agg = sentiment_data.groupby('datetime').agg({
    'score': 'mean',  # 平均情感分数
    'label': lambda x: x.mode()[0]  # 主要情感（出现最多的标签）
}).reset_index()

# 将情感标签映射为数值
sentiment_agg['label'] = sentiment_agg['label'].map({'positive': 1, 'neutral': 0, 'negative': -1})

# 合并数据
merged_data = stock_data.merge(sentiment_agg, left_on='Date', right_on='datetime', how='left')

# 填充缺失值（如果有日期的情感数据缺失）
merged_data['score'] = merged_data['score'].fillna(0)  # 缺失情感分数填充为0
merged_data['label'] = merged_data['label'].fillna(0)  # 缺失情感标签填充为0

merged_data = merged_data.drop(columns=['datetime'])

merged_data.to_csv("./stock/features_sentiment.csv", index=False)
