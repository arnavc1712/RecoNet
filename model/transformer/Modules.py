import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        # print("ATTENTION Q")
        # print(q)
        # print("ATTENTION K")
        # print(k)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        print("ATTENTION")
        print(attn)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        
        output = torch.bmm(attn, v)

        return output, attn
