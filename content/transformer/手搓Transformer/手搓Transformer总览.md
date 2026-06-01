# 手搓 Transformer 总览

手搓 Transformer 不应该一上来追求“完整复刻大模型”。

更合理的路线是：

```text
先跑通最小模块
再拼成 block
再跑通训练
最后再加工程细节
```

这套手搓笔记的目标是让你能回答：

```text
每一行代码对应论文里的哪个模块？
每个张量的 shape 是什么？
为什么这里要 transpose？
loss 为什么要 reshape？
mask 为什么要这样写？
```

## 推荐路线

### 1. 先手写 Attention

- [[手搓Attention|手搓 Attention]]

你要能手写：

```text
QK^T
scale
mask
softmax
乘 V
```

并且能说清楚 shape：

```text
q: [B, H, S, Dh]
k: [B, H, S, Dh]
scores: [B, H, S, S]
out: [B, H, S, Dh]
```

### 2. 再手写完整最小模型

- [[手搓最小Transformer|手搓最小 Transformer]]

这里不追求功能完整，而是用一个 decoder-only 小模型跑通：

```text
token id -> embedding -> blocks -> logits -> loss
```

### 3. 最后做训练实验

- [[../训练实验/最小训练实验|最小训练实验]]
- [[../训练实验/训练排错清单|训练排错清单]]

手搓模型必须能训练一个玩具任务。
否则只是代码能 forward，不代表你真的理解训练。

## 手搓时最容易错的点

- 忘记 `d_model % num_heads == 0`。
- `K` 没有 `transpose(-2, -1)`。
- `softmax` 维度写错。
- causal mask 方向反了。
- mask 形状不能广播。
- transpose 后忘记 `.contiguous()`。
- logits 和 labels 没有 shift。
- `CrossEntropyLoss` 前对 logits 先 softmax。
- 没有 `optimizer.zero_grad()`。
- 只 forward，没有 `loss.backward()` 和 `optimizer.step()`。

## 最小手搓目标

你最终要能写出：

```text
class MultiHeadAttention
class FeedForward
class TransformerBlock
class TinyGPT
训练循环
生成函数
```

能跑通就够了。

不要一开始就加：

- RoPE
- KV Cache
- FlashAttention
- GQA
- mixed precision
- 分布式训练

这些是后续优化，不是第一版手搓的核心。

## 一句话总结

手搓 Transformer 的核心不是背代码，而是把这条线写出来：

```text
embedding -> attention -> residual -> norm -> ffn -> logits -> loss -> backward -> step
```

每一步 shape 都对，训练 loss 能降，才算真的手搓成功。

