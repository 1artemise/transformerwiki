# 现代 Transformer 变体

原始 Transformer 很重要，但现在的大模型已经做了很多改动。

这些改动不是推翻 Transformer，而是在几个关键位置做工程优化：

```text
归一化位置
位置编码
注意力计算
FFN 结构
推理缓存
长上下文
```

## Pre-LN 和 Post-LN

原始论文常见写法是 Post-LN：

```text
x -> Sublayer -> Add -> LayerNorm
```

也就是：

```text
LayerNorm(x + Sublayer(x))
```

后来很多大模型用 Pre-LN：

```text
x -> LayerNorm -> Sublayer -> Add
```

也就是：

```text
x + Sublayer(LayerNorm(x))
```

为什么？

Pre-LN 通常更适合训练很深的模型。

直觉上，梯度可以更直接地通过残差路径往前传，训练稳定性更好。

## LayerNorm 和 RMSNorm

LayerNorm 会做：

```text
减均值
除标准差
再乘 gamma 加 beta
```

RMSNorm 更简单，通常不减均值，只按均方根缩放。

直觉：

```text
LayerNorm：把分布拉到均值 0、方差 1
RMSNorm：主要控制向量尺度
```

很多现代 LLM 使用 RMSNorm，因为它更省计算，效果也很好。

## 位置编码从绝对到相对

原始 Transformer 用固定 sin/cos 绝对位置编码。

后来出现很多位置编码方法：

- Learned Position Embedding
- Relative Position Bias
- RoPE
- ALiBi

核心问题都一样：

```text
让模型知道 token 的顺序和相对距离
```

## RoPE

RoPE 全称 Rotary Position Embedding。

它不是简单把位置向量加到 token embedding 上，而是在 Q、K 上做旋转。

直觉上：

```text
位置差会体现在 Q 和 K 的相对角度里
```

所以 RoPE 很适合 attention，因为 attention 本来就要算 Q 和 K 的相似度。

很多 LLM 都使用 RoPE 或它的变体。

## MHA、MQA、GQA

原始多头注意力是 MHA：

```text
每个 head 都有自己的 Q/K/V
```

推理时 KV Cache 很占显存，于是出现了 MQA 和 GQA。

### MQA

Multi-Query Attention：

```text
多个 Q head 共享一组 K/V
```

优点是 KV Cache 更小。
缺点是表达能力可能受影响。

### GQA

Grouped-Query Attention：

```text
把 Q head 分组，每组共享 K/V
```

它是 MHA 和 MQA 的折中。

很多现代大模型会用 GQA 来降低推理显存和带宽压力。

## FFN 到 SwiGLU

原始 Transformer FFN：

```text
Linear -> ReLU -> Linear
```

现代 LLM 常用 SwiGLU：

```text
两个线性分支
一个分支做门控
再相乘
```

直觉是：

```text
模型不仅学习“变成什么”，还学习“哪些信息该通过”
```

这增强了 FFN 的表达能力。

## FlashAttention

普通 attention 会显式生成很大的注意力矩阵：

```text
[seq_len, seq_len]
```

长序列时显存压力很大。

FlashAttention 的核心是：

```text
减少中间注意力矩阵的显存读写
用分块方式在 GPU 上更高效地算 attention
```

它不改变 attention 的数学结果，主要改变计算方式。

## 长上下文优化

长上下文的主要问题：

```text
attention 计算量随长度平方增长
KV Cache 随长度线性增长
```

常见优化：

- FlashAttention
- Sliding Window Attention
- Sparse Attention
- PagedAttention
- Prefix Caching
- RoPE scaling

这些技术都围绕一个目标：

```text
让模型能处理更长上下文，同时不把显存和延迟炸掉
```

## 现代 LLM 常见结构

一个现代 decoder-only LLM 通常像这样：

```text
Token Embedding
  ↓
N 层 Transformer Block
  ↓
RMSNorm
  ↓
Linear 到词表
```

每个 block 可能是：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

和原始 Transformer 相比，常见变化：

- 只用 decoder。
- 用 causal mask。
- 用 RoPE。
- 用 RMSNorm。
- 用 SwiGLU。
- 用 GQA/MQA。
- 推理时用 KV Cache。

## 一句话总结

现代 Transformer 不是完全新的东西。

它是在原始 Transformer 的主线上继续优化：

```text
更稳地训练
更快地推理
更省显存
更长上下文
更强表达能力
```

学原始 Transformer 是打底，理解这些变体才更接近现在的大模型工程。

