"""
Transformer模型的PyTorch实现
基于论文"Attention is All You Need" (https://arxiv.org/abs/1706.03762)
"""

import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    词嵌入层：将输入的词索引转换为词向量表示
    """
    def __init__(self, vocab_size, d_model):
        """
        初始化词嵌入层
        Args:
            vocab_size: 词汇表大小
            d_model: 词向量维度
        """
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入的词索引张量，形状为(batch_size, seq_len)
        Returns:
            词嵌入张量，形状为(batch_size, seq_len, d_model)
        """
        # 根据论文，词嵌入需要乘以sqrt(d_model)进行缩放
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码：为序列中的每个位置添加位置信息
    使用正弦和余弦函数生成位置编码
    """
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        Args:
            d_model: 词向量维度
            max_len: 序列最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 计算位置编码的分母部分
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # 偶数位置使用正弦函数编码
        pe[:, 0::2] = torch.sin(position / div_term)
        # 奇数位置使用余弦函数编码
        pe[:, 1::2] = torch.cos(position / div_term)

        # 增加批次维度并注册为buffer（不参与梯度更新）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)
        Returns:
            添加了位置编码的张量，形状为(batch_size, seq_len, d_model)
        """
        # 将位置编码加到输入张量上
        x = x + self.pe[:, :x.size(1), :]
        return x


def attention(query, key, value, mask=None, dropout=None):
    """
    计算注意力权重并应用到值向量上
    Args:
        query: 查询张量
        key: 键张量
        value: 值张量
        mask: 掩码张量，用于屏蔽某些位置
        dropout: Dropout层
    Returns:
        加权后的值向量和注意力权重
    """
    # 获取查询向量的维度
    d_k = query.size(-1)
    # 计算注意力分数：Q * K^T / sqrt(d_k)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    # 如果有掩码，将掩码位置的分数设为负无穷
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 对注意力分数进行softmax归一化
    attn = torch.softmax(scores, dim=-1)

    # 如果有dropout，对注意力权重应用dropout
    if dropout is not None:
        attn = dropout(attn)

    # 将注意力权重应用到值向量上
    return attn @ value, attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    将注意力计算分成多个头并行处理，最后将结果拼接
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        初始化多头注意力
        Args:
            h: 注意力头的数量
            d_model: 模型维度
            dropout: Dropout概率
        """
        super(MultiHeadedAttention, self).__init__()
        # 检查模型维度是否能被头数整除
        assert d_model % h == 0
        # 计算每个头的维度
        self.d_k = d_model // h
        self.h = h

        # 定义线性变换层
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 掩码张量
        Returns:
            多头注意力计算结果
        """
        batch_size = query.size(0)

        def transform(x, linear):
            """
            对输入进行线性变换并重塑形状
            """
            x = linear(x)
            # 重塑为(batch_size, seq_len, h, d_k)并转置为(batch_size, h, seq_len, d_k)
            return x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 对Q、K、V进行线性变换和形状重塑
        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        # 计算注意力
        x, _ = attention(query, key, value, mask, self.dropout)

        # 将多头结果拼接并进行线性变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linear_out(x)


class FeedForward(nn.Module):
    """
    前馈神经网络：两个线性变换之间加入ReLU激活函数
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络
        Args:
            d_model: 模型维度
            d_ff: 前馈网络中间层维度
            dropout: Dropout概率
        """
        super(FeedForward, self).__init__()
        # 定义前馈网络结构
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # 第一个线性层
            nn.ReLU(),                   # ReLU激活函数
            nn.Linear(d_ff, d_model),    # 第二个线性层
            nn.Dropout(dropout)          # Dropout层
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            前馈网络输出
        """
        return self.net(x)


class AddNorm(nn.Module):
    """
    残差连接和层归一化
    """
    def __init__(self, size, dropout=0.1):
        """
        初始化AddNorm模块
        Args:
            size: 输入张量的特征维度
            dropout: Dropout概率
        """
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)    # 层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, x, sublayer):
        """
        前向传播
        Args:
            x: 输入张量
            sublayer: 子层（注意力层或前馈网络）
        Returns:
            经过残差连接和层归一化的输出
        """
        # 残差连接：输出 = 输入 + 子层输出
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    编码器层：包含一个多头自注意力层和一个前馈网络层
    """
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        """
        初始化编码器层
        Args:
            d_model: 模型维度
            self_attn: 多头自注意力层
            feed_forward: 前馈网络层
            dropout: Dropout概率
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn          # 多头自注意力
        self.feed_forward = feed_forward    # 前馈网络
        # 两个AddNorm模块
        self.sublayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
             ]
        )

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入张量
            mask: 掩码张量
        Returns:
            编码器层输出
        """
        # 第一个子层：多头自注意力 + 残差连接和层归一化
        x = self.sublayers[0](x, lambda y: self.self_attn(x, x, x, mask))
        # 第二个子层：前馈网络 + 残差连接和层归一化
        return self.sublayers[1](x, self.feed_forward(x))


class DecoderLayer(nn.Module):
    """
    解码器层：包含两个多头注意力层和一个前馈网络层
    """
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout=0.1):
        """
        初始化解码器层
        Args:
            d_model: 模型维度
            self_attn: 多头自注意力层
            cross_attn: 多头交叉注意力层
            feed_forward: 前馈网络层
            dropout: Dropout概率
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn          # 多头自注意力（掩码）
        self.cross_attn = cross_attn        # 多头交叉注意力
        self.feed_forward = feed_forward    # 前馈网络
        # 三个AddNorm模块
        self.sublayers = nn.ModuleList(
            [
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout),
                AddNorm(d_model, dropout)
            ]
        )

    def forward(self, x, memory, src_mask=None, tag_mask=None):
        """
        前向传播
        Args:
            x: 解码器输入
            memory: 编码器输出
            src_mask: 源序列掩码
            tag_mask: 目标序列掩码
        Returns:
            解码器层输出
        """
        # 第一个子层：掩码多头自注意力 + 残差连接和层归一化
        out1 = self.sublayers[0](x, lambda y1: self.self_attn(x, x, x, tag_mask))
        # 第二个子层：多头交叉注意力 + 残差连接和层归一化
        out2 = self.sublayers[1](out1, lambda y2: self.cross_attn(out1, memory, memory, src_mask))
        # 第三个子层：前馈网络 + 残差连接和层归一化
        out3 = self.sublayers[2](out2, self.feed_forward(out2))
        return out3


class Transformer(nn.Module):
    """
    Transformer模型：包含编码器和解码器
    """
    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        """
        初始化Transformer模型
        Args:
            src_vocab: 源词汇表大小
            tag_vocab: 目标词汇表大小
            d_model: 模型维度
            N: 编码器和解码器层数
            h: 注意力头数
            d_ff: 前馈网络中间层维度
            dropout: Dropout概率
        """
        super(Transformer, self).__init__()

        # 源序列嵌入层（词嵌入 + 位置编码）
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model)
        )
        # 目标序列嵌入层（词嵌入 + 位置编码）
        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model),
            PositionalEncoding(d_model)
        )

        # 创建注意力和前馈网络的工厂函数
        attn = lambda: MultiHeadedAttention(h, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)

        # 创建编码器层列表（N层）
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)]
        )
        # 创建解码器层列表（N层）
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)]
        )

        # 输出层：将模型输出映射到目标词汇表
        self.out = nn.Linear(d_model, tag_vocab)

    def encode(self, src, src_mask):
        """
        编码器前向传播
        Args:
            src: 源序列输入
            src_mask: 源序列掩码
        Returns:
            编码器输出
        """
        # 对源序列进行嵌入处理
        x = self.src_embed(src)
        # 依次通过每个编码器层
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tag, memory, src_mask, tag_mask):
        """
        解码器前向传播
        Args:
            tag: 目标序列输入
            memory: 编码器输出
            src_mask: 源序列掩码
            tag_mask: 目标序列掩码
        Returns:
            解码器输出
        """
        # 对目标序列进行嵌入处理
        x = self.tag_embed(tag)
        # 依次通过每个解码器层
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tag_mask)
        return x

    def forward(self, src, tag, src_mask=None, tag_mask=None):
        """
        前向传播
        Args:
            src: 源序列输入
            tag: 目标序列输入
            src_mask: 源序列掩码
            tag_mask: 目标序列掩码
        Returns:
            模型输出（未经过softmax）
        """
        # 编码器处理源序列
        memory = self.encode(src, src_mask)
        # 解码器处理目标序列
        out = self.decode(tag, memory, src_mask, tag_mask)
        # 输出层映射到词汇表空间
        return self.out(out)