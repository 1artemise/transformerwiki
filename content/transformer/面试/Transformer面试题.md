# Transformer 面试题

这篇不是背答案用的，而是训练你怎么把 Transformer 讲清楚。

面试里最重要的不是把公式念出来，而是能说明：

```text
这个模块解决什么问题
为什么这样设计
代码里怎么实现
训练或推理时会出什么坑
```

## 1. Transformer 解决了 RNN 什么问题

**回答主线：**

RNN 需要按时间顺序一步步处理序列，后一个 token 依赖前一个 token 的隐藏状态，所以并行能力差；长序列里信息也容易逐步衰减。

Transformer 用 self-attention 让每个 token 直接和其他 token 建关系，可以并行计算整段序列，并且更容易建模长距离依赖。

可以补一句：

```text
Transformer 不是没有顺序信息，而是把顺序信息从递归结构里拿出来，交给位置编码处理。
```

## 2. Self-Attention 在做什么

**回答主线：**

Self-Attention 让同一句话里的 token 互相看。

每个 token 会根据 Q 和所有 token 的 K 算相关性，再用 softmax 得到权重，最后加权汇总 V。

公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

一句话：

```text
QK 算该看谁，softmax 算看多少，V 提供真正被拿走的信息。
```

## 3. Q、K、V 为什么要分开

**回答主线：**

Q、K、V 都来自同一个输入，但经过不同线性投影：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

分开的原因是：提问、匹配、提供信息是三种不同角色。

同一个 token 在不同角色下需要不同表示。

如果不分开，模型表达能力会弱很多。

## 4. 为什么要除以 sqrt(d_k)

**回答主线：**

`QK^T` 是点积，维度越大，点积结果的方差越大。

如果分数太大，softmax 会非常尖锐，接近 one-hot，梯度容易变小，训练不稳定。

除以 `sqrt(d_k)` 是为了缩放分数，让 softmax 不要过早饱和。

## 5. Multi-Head Attention 为什么有用

**回答主线：**

单个 attention 头只能在一个表示空间里建关系。

多头注意力把 `d_model` 拆到多个子空间，每个头可以学习不同类型的关系，比如：

```text
语法关系
指代关系
局部邻近关系
长距离依赖
```

最后把多个头拼接起来，再通过输出线性层融合。

代码形状：

```text
[B, S, D] -> [B, H, S, Dh] -> attention -> [B, S, D]
```

## 6. Masked Attention 为什么需要

**回答主线：**

Decoder 生成时不能看到未来 token。

训练时目标句子虽然完整存在，但预测当前位置时只能看当前位置之前的信息。

所以需要 causal mask 把未来位置挡住。

否则训练时模型会偷看答案，loss 很低，但推理时无法复现这个条件。

## 7. Encoder 和 Decoder 的区别

**回答主线：**

Encoder 负责理解输入，通常可以双向看完整输入。

Decoder 负责生成输出，需要 causal mask，只能看过去。

原始 Transformer 的 decoder 还有 cross-attention：

```text
Q 来自 decoder
K/V 来自 encoder
```

它让 decoder 生成时能查 encoder 对输入的理解。

## 8. BERT 和 GPT 的结构区别

**回答主线：**

BERT 是 encoder-only，适合理解任务。它可以双向看完整输入。

GPT 是 decoder-only，适合生成任务。它使用 causal mask，只能看当前位置之前的 token。

所以：

```text
BERT 更像读题
GPT 更像续写
```

## 9. 为什么 Transformer 需要位置编码

**回答主线：**

Self-attention 本身对 token 顺序不敏感。

如果不加位置编码：

```text
猫 追 狗
狗 追 猫
```

模型看到的是同一组 token，难以区分顺序导致的语义差异。

位置编码负责告诉模型 token 在哪里。

## 10. Add & Norm 的作用

**回答主线：**

Add 是残差连接：

```text
x + Sublayer(x)
```

它保留原始信息，并让梯度有更直接的回传路径，缓解深层网络训练困难。

Norm 通常是 LayerNorm，用来稳定每层输入输出的数值分布。

一句话：

```text
Add 保信息和梯度，Norm 稳数值分布。
```

## 11. LayerNorm 和 BatchNorm 有什么区别

**回答主线：**

BatchNorm 沿 batch 维统计，同一个特征在不同样本之间算均值方差。

LayerNorm 沿特征维统计，对单个 token 或样本内部的 hidden dim 算均值方差。

Transformer 常用 LayerNorm，因为 NLP 序列长度和 batch size 更不稳定，推理时 batch size 也可能很小。

## 12. FFN 在 Transformer 里有什么用

**回答主线：**

Attention 负责 token 之间交换信息。

FFN 负责每个 token 位置独立地做非线性变换。

通常结构：

```text
Linear(d_model, d_ff)
Activation
Linear(d_ff, d_model)
```

它提升模型表达能力，也占了 Transformer 很大一部分参数和计算。

## 13. Transformer 训练时和推理时有什么区别

**回答主线：**

训练时：

```text
前向传播 -> loss -> backward -> optimizer.step
```

权重会更新。

推理时：

```text
固定权重 -> 逐 token 生成
```

不会计算 loss，也不会更新权重。

Dropout 训练时开启，推理时关闭。

## 14. 为什么推理不能一次生成整句话

**回答主线：**

自回归模型生成第 `t` 个 token 时，需要前面已经生成的 token。

第 `t+1` 个 token 依赖第 `t` 个 token。

所以推理必须一步步生成。

训练时因为完整答案已知，可以用 mask 并行计算所有位置的 loss。

## 15. KV Cache 是什么

**回答主线：**

生成时历史 token 的 K/V 已经算过，而且不会变。

KV Cache 把每层的历史 K/V 存起来，下一个 token 只需要算新 token 的 Q/K/V，然后用 Q 去和缓存里的 K 做 attention。

它减少重复计算，用显存换速度。

补形状：

```text
K_cache: [B, H, L, Dh]
V_cache: [B, H, L, Dh]
```

## 16. 为什么长上下文很贵

**回答主线：**

Self-attention 的分数矩阵是：

```text
[seq_len, seq_len]
```

所以计算和显存会随序列长度平方增长。

推理时 KV Cache 也会随着上下文长度线性增长。

长上下文优化本质是在处理 attention 和 KV Cache 的成本。

## 17. FlashAttention 解决什么

**回答主线：**

FlashAttention 不改变 attention 数学公式。

它优化的是 GPU 上 attention 的计算方式，减少中间 attention 矩阵的显存读写，用分块方式提高速度和节省显存。

一句话：

```text
数学等价，工程更高效。
```

## 18. RoPE 是什么

**回答主线：**

RoPE 是旋转位置编码。

它不是把位置向量直接加到 embedding 上，而是在 Q/K 上注入位置信息。

这样位置差可以体现在 Q/K 的相对角度里，很适合 attention。

## 19. MHA、MQA、GQA 区别

**回答主线：**

MHA：每个 query head 都有自己的 K/V。

MQA：多个 query head 共享一组 K/V，KV Cache 更省。

GQA：query head 分组，每组共享 K/V，是 MHA 和 MQA 的折中。

现代 LLM 常用 GQA 来平衡效果和推理成本。

## 20. 面试手写 attention 要注意什么

核心代码：

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
scores = scores.masked_fill(mask == 0, float("-inf"))
attn = torch.softmax(scores, dim=-1)
out = attn @ v
```

要主动说明：

- `transpose(-2, -1)` 转的是 K 的最后两维。
- `softmax(dim=-1)` 表示对“看谁”的维度归一化。
- mask 要能广播到 scores。
- q/k/v 通常是 `[B, H, S, Dh]`。

## 21. loss 不下降怎么排查

按顺序说：

```text
数据和 label 是否正确
logits/labels shape 是否正确
是否错开一位预测 next token
mask 是否写反
loss 是否忽略 padding
梯度是否存在
权重是否更新
学习率是否合适
```

不要直接说“调参”。
先把训练链路排干净。

## 22. 如果让你从零实现 Transformer，你怎么拆

回答结构：

```text
Embedding
Positional Encoding
Scaled Dot-Product Attention
Multi-Head Attention
Feed Forward
Add & LayerNorm
EncoderLayer
DecoderLayer 或 Decoder-only Block
Output Linear
Training Loop
```

然后强调先跑通最小任务，不要一开始就上大语料。

## 23. 面试中容易加分的表达

- “Attention 权重不是人工规则，是 W_Q/W_K/W_V 训练出来后产生的。”
- “训练改变权重，解码策略只改变选 token 的方式。”
- “Transformer 的很多工程优化都围绕两件事：attention 的平方复杂度和 KV Cache 的显存。”
- “我会先用玩具任务验证训练链路，再上真实数据。”
- “排查 loss 不降时，我会先看 shape、mask、label shift、grad norm，而不是直接换模型。”

## 24. 一分钟总结版

Transformer 用 self-attention 替代 RNN 的顺序递归，让每个 token 能直接和其他 token 建关系，并且训练时更容易并行。

它的核心模块是 multi-head attention、FFN、残差连接和 LayerNorm。

训练时通过 loss 和反向传播更新权重；推理时权重固定，自回归逐 token 生成。

现代 LLM 在这个基础上加入 RoPE、RMSNorm、GQA、SwiGLU、KV Cache、FlashAttention 等优化，主要目标是更稳定训练、更高效推理、更长上下文。

