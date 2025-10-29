# Transformer 模型详解

本项目是基于论文《Attention Is All You Need》的 Transformer 模型的 PyTorch 实现。本文档将逐步解释每一部分的代码，帮助初学者理解 Transformer 的工作机制。

## 什么是 Transformer？

Transformer 是一种用于自然语言处理任务的深度学习模型架构，首次出现在 2017 年的论文《Attention Is All You Need》中。它完全基于注意力机制，摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），在机器翻译等序列到序列的任务中表现出色。

## 整体架构

Transformer 由两大部分组成：
1. **编码器（Encoder）**：将输入序列（如源语言句子）转换为一系列隐藏表示
2. **解码器（Decoder）**：根据编码器的输出和已生成的部分输出序列，生成目标序列（如翻译后的句子）

两者都基于"编码器-解码器"结构，但内部实现完全不同与传统的 RNN 或 CNN。

## 代码详解

### 1. 导入依赖

```python
import torch
import torch.nn as nn
import math
```

我们导入了 PyTorch 核心库和数学库，用于构建神经网络和执行数学运算。

- `torch`: PyTorch 核心库，提供了张量计算和自动微分功能
- `torch.nn`: PyTorch 的神经网络模块，包含各种预定义的层和函数
- `math`: Python 标准库中的数学模块，用于数学常数和函数

### 2. 词嵌入（Embeddings）

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
```

这部分负责将输入的单词索引转换为稠密向量表示（词嵌入）。
- `vocab_size`: 词汇表大小
- `d_model`: 词向量维度
- 注意：根据论文，词嵌入后需要乘以 `sqrt(d_model)` 进行缩放

涉及的 PyTorch 函数详解：
- `nn.Embedding(vocab_size, d_model)`: 创建一个简单的查找表，将索引映射到词向量
  - `vocab_size`: 词汇表大小，即有多少个不同的词
  - `d_model`: 词向量维度，即每个词用多少维向量表示
  - 这个层在训练过程中会学习到每个词的向量表示

- `math.sqrt(self.d_model)`: 计算 d_model 的平方根，用于缩放词嵌入向量
  - 这是论文中的建议，用于控制嵌入向量的大小，使位置编码更稳定

### 3. 位置编码（Positional Encoding）

```python
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
```

由于 Transformer 不使用 RNN 或 CNN，它没有内置的序列顺序概念。位置编码给模型提供了单词在句子中位置的信息。
- 使用正弦和余弦函数生成不同位置的编码
- 这样可以让模型学会关注相对位置和距离关系

涉及的 PyTorch 函数详解：
- `torch.zeros(max_len, d_model)`: 创建一个指定形状的全零张量
  - `max_len`: 最大序列长度
  - `d_model`: 模型维度
  - 用于初始化位置编码矩阵

- `torch.arange(0, max_len)`: 创建一个从 0 到 max_len-1 的一维张量
  - 用于生成位置索引

- `unsqueeze(1)`: 在指定维度增加一个维度
  - 将一维张量 (max_len,) 变为二维张量 (max_len, 1)
  - 便于后续的广播运算

- `torch.exp()`: 计算自然指数函数
  - 用于计算位置编码中的分母部分

- `torch.sin()` 和 `torch.cos()`: 正弦和余弦函数
  - 用于生成位置编码，正弦用于偶数位置，余弦用于奇数位置
  - 这种设计使得模型可以学习到相对位置信息

- `register_buffer('pe', pe)`: 注册一个缓冲区
  - 缓冲区中的张量不会被优化器更新，但会包含在模型的状态字典中
  - 适用于不需要梯度更新但需要保存的张量，如位置编码

### 4. 注意力机制（Attention）

```python
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    return attn @ value, attn
```

注意力机制是 Transformer 的核心。其基本思想是对于序列中的每个元素，计算它与其他所有元素的相关性得分，然后根据这些得分加权组合所有元素的值。
- `Query`, `Key`, `Value` 是注意力机制中的三个概念
- 公式：`Attention(Q,K,V) = softmax(QK^T/√dk)V`

涉及的 PyTorch 函数详解：
- `query.size(-1)`: 获取张量最后一个维度的大小
  - 在注意力机制中，这通常是每个头的维度 d_k

- `@`: 矩阵乘法运算符
  - 等价于 `torch.matmul()`，用于计算 Q 和 K 的点积

- `transpose(-2, -1)`: 交换张量的最后两个维度
  - 用于计算 K 的转置，使得矩阵乘法可以正确进行

- `masked_fill(mask == 0, float('-inf'))`: 用指定值填充掩码位置
  - 将掩码为 0 的位置填充为负无穷
  - 在后续的 softmax 操作中，这些位置的注意力权重会趋近于 0

- `torch.softmax(scores, dim=-1)`: 对张量沿指定维度进行 softmax 归一化
  - 将注意力得分转换为概率分布
  - 每行的和为 1，表示每个位置对其他位置的注意力权重

### 5. 多头注意力（Multi-Head Attention）

```python
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
```

多头注意力允许模型在不同的表示子空间中同时关注信息的不同方面。
- 将 Query, Key, Value 通过线性变换投影到 `h` 个不同的子空间
- 在每个子空间分别执行注意力计算
- 最后将结果拼接并进行线性变换

涉及的 PyTorch 函数详解：
- `assert d_model % h == 0`: 断言检查
  - 确保模型维度可以被头数整除，否则无法均匀分割

- `nn.Linear(d_model, d_model)`: 全连接层（线性变换）
  - 对输入进行线性变换：`y = xA^T + b`
  - 用于将 Q、K、V 投影到不同的子空间

- `view(batch_size, -1, self.h, self.d_k)`: 改变张量形状
  - `-1` 表示该维度大小自动推断
  - 将线性变换后的张量重塑为 (batch_size, seq_len, h, d_k)

- `transpose(1, 2)`: 交换第1和第2维度
  - 将张量形状从 (batch_size, seq_len, h, d_k) 变为 (batch_size, h, seq_len, d_k)
  - 使得每个头可以独立进行计算

- `contiguous()`: 返回一个内存连续的张量
  - 在执行 view 操作前，确保张量在内存中是连续存储的
  - 某些张量操作（如 transpose）可能导致内存不连续

### 6. 前馈神经网络（Feed Forward）

```python
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
```

这是一个简单的两层全连接网络，应用于每个位置的表示：
- 第一层将维度从 `d_model` 扩展到 `d_ff`
- 使用 ReLU 激活函数
- 第二层将维度从 `d_ff` 压缩回 `d_model`

涉及的 PyTorch 函数详解：
- `nn.Sequential`: 顺序容器
  - 按顺序执行其中的模块
  - 简化了前向传播的代码

- `nn.ReLU()`: ReLU 激活函数
  - `f(x) = max(0, x)`
  - 引入非线性，增强模型表达能力

- `nn.Dropout(dropout)`: Dropout 层
  - 在训练过程中随机将部分神经元输出设为 0
  - 用于防止过拟合，提高模型泛化能力

### 7. 残差连接和层归一化（Add & Norm）

```python
class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

这是 Transformer 中的一个重要设计模式：
- Layer Normalization: 对每层的输入进行标准化
- Residual Connection: 将标准化后的输入与子层输出相加
- 这种设计有助于训练深层网络，缓解梯度消失问题

涉及的 PyTorch 函数详解：
- `nn.LayerNorm(size)`: 层归一化
  - 对每个样本的特征维度进行归一化
  - 与批量归一化不同，层归一化不依赖于批次大小
  - 计算公式：`y = (x - E[x]) / sqrt(Var[x] + ε) * γ + β`

### 8. 编码器层（Encoder Layer）

```python
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
```

编码器层由两个子层组成：
1. 多头自注意力机制（Self-Attention）
2. 前馈神经网络

每个子层都使用残差连接和层归一化。

涉及的 PyTorch 函数详解：
- `nn.ModuleList`: 模块列表
  - 存储子模块的列表，与普通列表不同，其中的模块会自动注册到网络中
  - 便于管理和访问网络中的子模块

- `lambda y: self.self_attn(x, x, x, mask)`: Lambda 表达式
  - 创建一个匿名函数，用于适配 AddNorm 接口
  - 在 AddNorm 中，sublayer 函数只接受一个参数，但注意力机制需要多个参数

### 9. 解码器层（Decoder Layer）

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
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
```

解码器层比编码器层更复杂，由三个子层组成：
1. 掩码多头自注意力机制（Masked Self-Attention）
2. 多头交叉注意力机制（Cross-Attention）
3. 前馈神经网络

掩码是为了防止在预测第 `i` 个位置时看到后续位置的信息。

### 10. 完整的 Transformer 模型

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model)
        )
        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model),
            PositionalEncoding(d_model)
        )

        attn = lambda: MultiHeadedAttention(h, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)]
        )

        self.out = nn.Linear(d_model, tag_vocab)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tag, memory, src_mask, tag_mask):
        x = self.tag_embed(tag)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tag_mask)
        return x

    def forward(self, src, tag, src_mask=None, tag_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tag, memory, src_mask, tag_mask)
        return self.out(out)
```

完整模型包括：
- 源序列和目标序列的嵌入层（词嵌入+位置编码）
- N 层编码器
- N 层解码器
- 输出投影层

涉及的 PyTorch 函数详解：
- `nn.Sequential`: 顺序容器
  - 按顺序执行其中的模块
  - 用于组合嵌入层和位置编码层

- `nn.ModuleList([... for _ in range(N)])`: 使用列表推导式创建 N 个模块
  - 用于创建 N 层编码器和解码器
  - 每一层都是独立的模块实例

## 总结

Transformer 模型的主要优势：
1. **并行化**：不像 RNN 需要按顺序处理序列，Transformer 可以并行处理所有位置
2. **长距离依赖**：注意力机制可以直接建模序列中任意两个位置的关系
3. **可解释性**：注意力权重提供了一定程度的可解释性

虽然最初是为机器翻译设计的，但 Transformer 已经广泛应用于各种 NLP 任务，并衍生出了 BERT、GPT 等著名模型。