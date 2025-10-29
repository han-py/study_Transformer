import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    return attn @ value, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear):
            x = linear(x)
            return x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        x, _ = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linear_out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
             ]
        )

    def forward(self, x, mask=None):
        x = self.sublayers[0](x, lambda y: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward(x))


class Encoder(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout=0.1):
        super(Encoder, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
            ]
        )

    def forward(self, x, memory, src_mask=None, tag_mask=None):
        out1 = self.sublayers[0](x, lambda y1: self.self_attn(x, x, x, tag_mask))
        out2 = self.sublayers[1](out1, lambda y2: self.cross_attn(out1, memory, memory, src_mask))
        out3 = self.sublayers[2](out2, self.feed_forward(out2))
        return out3