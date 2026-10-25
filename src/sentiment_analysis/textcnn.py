"""基础 TextCNN 情感分类模型。"""
import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, num_filters: int,
                 kernel_size: int, padding_index: int, dropout: float = 0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=padding_index)
        self.convolution = nn.Conv1d(embedding_dim, num_filters, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, 2)

    def forward(self, token_ids, lengths):
        if (lengths <= 0).any():
            raise ValueError("Each sequence must contain at least one token.")

        embeddings = self.embedding(token_ids)
        convolution_input = embeddings.transpose(1, 2)
        convolution_output = torch.relu(self.convolution(convolution_input))

        # 最大池化时排除完全由填充组成的窗口
        positions = torch.arange(convolution_output.shape[2], device=token_ids.device)
        valid_windows = positions.unsqueeze(0) < lengths.to(token_ids.device).unsqueeze(1)
        masked_output = convolution_output.masked_fill(~valid_windows.unsqueeze(1), float("-inf"))
        pooled_output = masked_output.max(dim=2).values

        return self.classifier(self.dropout(pooled_output))
