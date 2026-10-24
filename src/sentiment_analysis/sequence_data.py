"""序列模型的数据准备。"""
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str) -> list[str]:
    """使用与 TF-IDF 一致的小写化和分词规则。"""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts, max_words: int = 20_000) -> dict[str, int]:
    """按训练集词频建立词表，频率相同时按词项排序。"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(tokenize(text))

    sorted_words = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for word, count in sorted_words[:max_words]:
        vocabulary[word] = len(vocabulary)

    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int]) -> list[int]:
    """将评论转换为 Token ID，词表外的词项映射到 UNK。"""
    return [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokenize(text)]


def prepare_sequence(text: str, vocabulary: dict[str, int], max_length: int) -> tuple[list[int], int]:
    """截断或填充序列，并返回填充前的有效长度。"""
    token_ids = encode_text(text, vocabulary)[:max_length]
    sequence_length = len(token_ids)
    token_ids += [vocabulary[PAD_TOKEN]] * (max_length - sequence_length)

    return token_ids, sequence_length


class SentimentDataset(Dataset):
    def __init__(self, data: pd.DataFrame, vocabulary: dict[str, int], max_length: int):
        self.texts = data["text"].reset_index(drop=True)
        self.labels = data["label"].reset_index(drop=True)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        token_ids, sequence_length = prepare_sequence(
            self.texts.iloc[index], self.vocabulary, self.max_length,
        )
        label = self.labels.iloc[index]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(sequence_length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )
