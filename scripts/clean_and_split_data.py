"""
Clean and split data into train data, validation data and test data.
"""
# 数据清理
from sentiment_analysis.paths import RAW_DIR, CLEANED_DIR
from sklearn.model_selection import train_test_split
import pandas as pd
import re

# 加载原始数据集
train_raw = pd.read_csv(RAW_DIR / "train.csv")
test_raw = pd.read_csv(RAW_DIR / "test.csv")

# 对原始数据集去重
train_unique = train_raw.drop_duplicates(subset= "text").reset_index(drop= True)
test_unique = test_raw.drop_duplicates(subset= "text").reset_index(drop= True)

# 去除 test 中与 train 重复的 text 数据
test_clean = test_unique[
    ~test_unique["text"].isin(train_unique["text"])
].reset_index(drop=True)

# 划分训练集与验证集
train_clean, validation_clean = train_test_split(
    train_unique,
    test_size=0.1,
    random_state=42,                    # 固定随机数种子，保证分层结果可复现
    stratify=train_unique["label"],     # 根据标签进行分层抽样
)

train_clean = train_clean.reset_index(drop=True)
validation_clean = validation_clean.reset_index(drop=True)

splits = {
    "train": train_clean,
    "validation": validation_clean,
    "test": test_clean,
}

def clean_html(text: str) -> str:
    """仅移除标准的 html 标签。"""

    # 将换行标签替换为空格
    text = re.sub(
        r"</?(?:br|hr|p)\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 移除倾斜加粗等标签
    text = re.sub(
        r"</?(?:i|em)\s*>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text

# 清洗数据
for data in splits.values():
    data["text"] = data["text"].apply(clean_html)


# 保存数据
CLEANED_DIR.mkdir(parents= True, exist_ok= True)

train_clean.to_csv(CLEANED_DIR / "train.csv", index= False)
validation_clean.to_csv(CLEANED_DIR / "val.csv", index= False)
test_clean.to_csv(CLEANED_DIR / "test.csv", index= False)