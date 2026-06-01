# Attention Is All You Need 精读

这篇不是逐句翻译，而是按学习 Transformer 最需要理解的顺序读论文。

论文原文：Vaswani et al., 2017, **Attention Is All You Need**。

## 论文想解决的问题

在这篇论文之前，机器翻译主流方法依赖 RNN、LSTM 或 CNN。

RNN 的问题是：

```text
必须按时间顺序一步步处理
很难并行
长距离依赖容易变弱
```

论文的核心观点是：

```text
不用循环结构，也不用卷积结构，只用注意力机制就可以建一个强大的序列模型。
```

这也是标题里 “Attention Is All You Need” 的意思。

## 原始 Transformer 是什么结构

原始 Transformer 是 encoder-decoder 架构。

```text
Encoder：读输入句子
Decoder：生成输出句子
```

一共有：

```text
6 层 Encoder
6 层 Decoder
```

每层 Encoder：

```text
Multi-Head Self-Attention
Add & Norm
Feed Forward
Add & Norm
```

每层 Decoder：

```text
Masked Multi-Head Self-Attention
Add & Norm
Encoder-Decoder Attention
Add & Norm
Feed Forward
Add & Norm
```

## 为什么不用 RNN

论文最实际的理由是并行。

RNN 处理句子：

```text
第1个词 -> 第2个词 -> 第3个词 -> ...
```

后一个词依赖前一个词的隐藏状态，所以很难完全并行。

Transformer 处理句子：

```text
所有 token 同时进入 self-attention
```

每个 token 都能直接和其他 token 建立关系。

这让训练更适合 GPU。

## Scaled Dot-Product Attention

论文里的注意力公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

拆开看：

```text
QK^T：算相关性
/ sqrt(d_k)：控制分数尺度
softmax：变成权重
乘 V：加权汇总信息
```

最关键的不是公式长什么样，而是它让每个 token 都能问：

```text
我应该从其他 token 那里拿多少信息？
```

对应笔记：[[../关键模块/注意力机制|注意力机制]]

## 为什么要 scale

如果 `d_k` 很大，点积结果容易变大。

分数太大时，softmax 会变得很尖锐：

```text
[20, 1, 0] -> [几乎 1, 几乎 0, 几乎 0]
```

这样训练早期模型会过快把注意力压到少数位置，梯度也容易变得不稳定。

所以论文除以：

```text
sqrt(d_k)
```

目的就是让 softmax 的输入不要太极端。

## Multi-Head Attention

论文没有只做一个注意力，而是做多个头。

直觉上：

```text
一个头学主谓关系
一个头学动宾关系
一个头学位置关系
一个头学指代关系
```

当然真实模型里每个头不一定这么清晰，但多头的意义就是让模型在多个子空间里同时建关系。

论文 base 模型里：

```text
d_model = 512
num_heads = 8
d_k = 64
```

每个头算完后拼接，再过一个输出线性层。

## Position-wise Feed Forward

每层 attention 后面还有 FFN。

公式大概是：

```text
FFN(x) = max(0, xW1 + b1)W2 + b2
```

它对每个位置单独作用，不在 token 之间交换信息。

所以可以这样分工：

```text
Attention：token 之间交流
FFN：每个 token 自己加工
```

论文 base 模型里：

```text
d_model = 512
d_ff = 2048
```

也就是先升维，再降回原维度。

## Positional Encoding

因为 Transformer 不按顺序读，所以必须额外加位置信息。

论文使用固定 sin/cos 位置编码：

```text
PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

这个设计让不同位置有不同的位置向量，同时保留一定相对位置信息。

对应笔记：[[../关键模块/位置编码|位置编码]]

## Masked Self-Attention

Decoder 生成时不能看未来。

训练时虽然完整目标句子都在数据里，但模型预测第 `t` 个 token 时，只能看：

```text
1 到 t-1 的 token
```

所以 decoder 的 self-attention 要加 mask。

否则模型训练时会偷看答案，loss 会很好看，但真正生成时会崩。

## Label Smoothing

论文还用了 label smoothing。

普通交叉熵会要求模型对正确答案非常自信：

```text
正确词：1.0
其他词：0.0
```

Label smoothing 会把目标分布变软一点：

```text
正确词：0.9
其他词：分一点点概率
```

这样模型不会过度自信，泛化通常更好。

对应笔记：[[../前置知识/正则化|正则化]]

## 学习率 warmup

论文使用了 Adam，并配合 warmup 学习率策略。

训练初期学习率先升高，后面再下降。

直觉是：

```text
刚开始参数很乱，不要一上来走太猛
稳定后提高学习速度
后期接近较好区域，再慢慢减小步子
```

这也是 Transformer 训练稳定性的一个重要细节。

## 这篇论文真正重要的点

不是“提出一个新公式”这么简单。

它真正证明了：

```text
序列建模不一定需要循环结构
注意力可以作为主干结构
并行训练可以显著提升效率
```

后来的 BERT、GPT、T5、ViT 都是在这个底座上变化。

## 读论文时不要忽略的细节

- 原始 Transformer 是 encoder-decoder，不是 GPT 那种 decoder-only。
- self-attention 的计算量和序列长度平方相关。
- Add & Norm 对训练稳定性非常关键。
- FFN 占了大量参数和计算。
- 位置编码是为了解决 attention 不感知顺序的问题。
- mask 是为了让训练过程符合生成过程。
- 论文里的很多超参数不是理论唯一答案，而是工程实验下的选择。

## 一句话总结

这篇论文的主线是：

```text
用 self-attention 替代循环结构
用 multi-head 提供多个关系视角
用 position encoding 补顺序
用 Add & Norm 稳住深层训练
用 FFN 增强每个位置的非线性表达
```

理解这条线，再看 Transformer 架构图就不会觉得它只是模块堆叠。

