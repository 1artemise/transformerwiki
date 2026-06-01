# 手搓最小 Transformer

这一篇写一个最小 decoder-only Transformer。

它不是工业级 GPT，但可以跑通：

```text
输入 token
经过多头注意力和 FFN
输出词表 logits
计算 loss
反向传播更新权重
```

## 先定目标

我们实现这些模块：

```text
CausalSelfAttention
FeedForward
Block
TinyGPT
训练一步
生成函数
```

为了手搓清楚，先不加：

- RoPE
- KV Cache
- FlashAttention
- GQA
- mixed precision

## 完整代码

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=128,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx, targets=None):
        B, S = idx.shape
        assert S <= self.max_len

        pos = torch.arange(0, S, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
```

## 每个模块对上什么

### CausalSelfAttention

对应 decoder-only 的 masked self-attention。

它做：

```text
X -> QKV -> 多头 attention -> 输出投影
```

### FeedForward

对应 Transformer block 里的 FFN。

它对每个 token 位置独立做非线性变换。

### Block

这里用的是 Pre-LN：

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

现代 LLM 常用这种结构，因为深层训练更稳定。

### TinyGPT

这是一个最小 decoder-only 模型。

```text
token embedding
position embedding
多个 block
final LayerNorm
linear 到词表
```

## 最小训练一步

```python
vocab_size = 50
model = TinyGPT(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

B, S = 16, 32
x = torch.randint(0, vocab_size, (B, S))
y = torch.roll(x, shifts=-1, dims=1)

logits, loss = model(x, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(loss.item())
```

这只是玩具任务。
目标是确认：

```text
forward 能跑
loss 能算
backward 有梯度
step 能更新权重
```

## 检查 shape

在 `forward` 里可以临时打印：

```python
print("idx", idx.shape)
print("x", x.shape)
print("logits", logits.shape)
```

应该看到：

```text
idx: [B, S]
x: [B, S, D]
logits: [B, S, vocab_size]
```

Attention 内部：

```text
q/k/v: [B, H, S, Dh]
scores: [B, H, S, S]
```

## 为什么这里用 qkv 一个线性层

我们可以分别写：

```python
self.q = nn.Linear(D, D)
self.k = nn.Linear(D, D)
self.v = nn.Linear(D, D)
```

也可以写成一个：

```python
self.qkv = nn.Linear(D, 3 * D)
q, k, v = qkv.chunk(3, dim=-1)
```

后者更常见，因为一次线性层算出 QKV 更高效。

## 为什么 generate 里只取最后一个 logits

模型每次输出：

```text
[B, S, vocab_size]
```

推理时我们只关心最后一个位置预测的下一个 token：

```python
logits = logits[:, -1, :]
```

因为前面位置的预测已经没用了。

## 这个版本缺什么

它能帮助学习，但还不是完整工程模型。

缺少：

- tokenizer
- 真实数据集
- padding mask
- KV Cache
- 权重初始化细节
- 学习率 warmup
- mixed precision
- checkpoint 保存
- 评估指标

这些可以后面逐步加。

## 一句话总结

这个最小模型把 Transformer 的主线写出来了：

```text
token -> embedding -> masked attention -> residual -> norm -> ffn -> logits -> loss
```

只要它能在玩具任务上 loss 下降，你就已经真正跨过“只会看架构图”的阶段。

