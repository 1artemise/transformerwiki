# 手搓 Attention

这一篇只手写 attention。

先不管完整 Transformer，只把最核心的公式写成代码：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

## 输入形状

我们假设输入已经被拆成多头：

```text
q: [B, H, S, Dh]
k: [B, H, S, Dh]
v: [B, H, S, Dh]
```

含义：

```text
B：batch size
H：head 数量
S：序列长度
Dh：每个 head 的维度
```

## 最小实现

```python
import math
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn
```

这就是 attention 最核心的代码。

## 每一行在做什么

### 1. 算分数

```python
scores = q @ k.transpose(-2, -1)
```

形状：

```text
q:             [B, H, S, Dh]
k.transpose:   [B, H, Dh, S]
scores:        [B, H, S, S]
```

`scores[i, h, a, b]` 表示：

```text
第 i 个样本里，第 h 个头中，第 a 个 token 看第 b 个 token 的分数
```

### 2. scale

```python
scores = scores / math.sqrt(d_k)
```

避免分数太大，让 softmax 不要太尖锐。

### 3. mask

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

mask 为 0 的位置会被设成负无穷。

softmax 后这些位置概率就是 0。

### 4. softmax

```python
attn = torch.softmax(scores, dim=-1)
```

`dim=-1` 表示对最后一维归一化。

也就是对“当前 token 看所有 token 的分数”做 softmax。

### 5. 加权 V

```python
out = attn @ v
```

形状：

```text
attn: [B, H, S, S]
v:    [B, H, S, Dh]
out:  [B, H, S, Dh]
```

## causal mask 怎么写

decoder-only 模型不能看未来。

```python
def causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)
```

如果 `seq_len = 4`：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

第 3 个 token 可以看第 1、2、3 个 token，但不能看第 4 个。

## 测试一下

```python
B, H, S, Dh = 2, 4, 5, 8
q = torch.randn(B, H, S, Dh)
k = torch.randn(B, H, S, Dh)
v = torch.randn(B, H, S, Dh)

mask = causal_mask(S)
out, attn = scaled_dot_product_attention(q, k, v, mask)

print(out.shape)
print(attn.shape)
```

输出应该是：

```text
torch.Size([2, 4, 5, 8])
torch.Size([2, 4, 5, 5])
```

## 常见错误

### 1. K 转置错维度

应该是：

```python
k.transpose(-2, -1)
```

不要写成转 batch 或 head 维度。

### 2. softmax 维度错

应该对最后一维：

```python
torch.softmax(scores, dim=-1)
```

因为最后一维代表“看哪些 token”。

### 3. mask 方向反了

正确 causal mask 是下三角。

如果写成上三角，模型会看未来，看不到过去。

### 4. 全部被 mask

如果某一行全是 `-inf`，softmax 可能出 NaN。

要保证每个位置至少能看自己。

## 一句话总结

手搓 attention 只要抓住三件事：

```text
QK^T 得到 [S, S] 分数矩阵
softmax 在最后一维算注意力权重
权重乘 V 得到新的 token 表示
```

其他复杂模块，都是在这个核心上继续包装。

