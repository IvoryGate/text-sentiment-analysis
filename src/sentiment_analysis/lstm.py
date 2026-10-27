"""基础 LSTM 情感分类模型。"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_size: int,
                 padding_index: int, dropout: float = 0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=padding_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, token_ids, lengths):
        if (lengths <= 0).any():
            raise ValueError("Each sequence must contain at least one token.")

        embeddings = self.embedding(token_ids)
        packed_embeddings = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed_embeddings)
        final_hidden = hidden[-1]

        return self.classifier(self.dropout(final_hidden))


class BidirectionalLSTMClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_size: int,
                 padding_index: int, dropout: float = 0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=padding_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, token_ids, lengths):
        if (lengths <= 0).any():
            raise ValueError("Each sequence must contain at least one token.")

        embeddings = self.embedding(token_ids)
        packed_embeddings = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed_embeddings)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        return self.classifier(self.dropout(final_hidden))
