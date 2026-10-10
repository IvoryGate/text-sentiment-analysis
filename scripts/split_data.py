"""
Split data into train data, validation data and test data.
"""
# 划分训练集，测试集，验证集
from sklearn.model_selection import train_test_split
from sentiment_analysis.paths import RAW_DIR,INTERIM_DIR
import pandas as pd

train_raw = pd.read_csv(RAW_DIR / "train.csv")
test_raw = pd.read_csv(RAW_DIR / "test.csv")

train_df, val_df = train_test_split(
    train_raw,
    test_size= 0.1,
    train_size= 0.9,
    random_state= 42,               # 固定随机数种子，保证分层结果可复现
    stratify=train_raw["label"],    # 根据标签进行分层抽样
)

# 保存划分后的数据集
INTERIM_DIR.mkdir(parents= True, exist_ok= True)

train_df.to_csv(INTERIM_DIR / "train.csv", index= False, encoding= "utf-8")
val_df.to_csv(INTERIM_DIR / "val.csv", index= False, encoding= "utf-8")
test_raw.to_csv(INTERIM_DIR / "test.csv", index= False, encoding= "utf-8")