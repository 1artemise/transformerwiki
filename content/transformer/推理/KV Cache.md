# KV Cache

KV Cache 是大模型推理里非常重要的优化。

一句话理解：

**已经算过的 K 和 V 不要每一步都重复算，缓存起来，下次直接用。**

## 为什么需要 KV Cache

Decoder-only 模型生成文本时，是一个 token 一个 token 生成的。

比如已经生成：

```text
我 喜欢 吃
```

下一步要预测：

```text
苹果
```

如果不用 KV Cache，每一步都要重新计算整个序列的 attention。

生成第 4 个 token 时，重新算前 3 个 token。
生成第 5 个 token 时，重新算前 4 个 token。
生成第 1000 个 token 时，重新算前 999 个 token。

这很浪费。

## Attention 里哪些东西可以缓存

Self-Attention 每层会算：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

生成新 token 时，旧 token 的 K 和 V 不会变。

因为模型权重固定，旧 token 的表示在当前层里已经算过。

所以可以缓存：

```text
past_key
past_value
```

新 token 来了，只需要算新 token 的 Q、K、V，然后把新的 K/V 拼到缓存后面。

## 不用 KV Cache

每一步输入完整上下文：

```text
step 1: [A]
step 2: [A, B]
step 3: [A, B, C]
step 4: [A, B, C, D]
```

每次都重新算所有 token 的 K/V。

## 使用 KV Cache

每一步只处理新 token：

```text
step 1: 算 A 的 K/V，缓存
step 2: 算 B 的 K/V，拼到缓存
step 3: 算 C 的 K/V，拼到缓存
step 4: 算 D 的 K/V，拼到缓存
```

当前 token 的 Q 会和缓存里的所有 K 做 attention。

```text
Q_new @ K_cache^T
```

然后用权重汇总：

```text
attention_weight @ V_cache
```

## 形状怎么变

假设：

```text
B = batch
H = heads
Dh = head_dim
L = 已有上下文长度
```

缓存通常是：

```text
K_cache: [B, H, L, Dh]
V_cache: [B, H, L, Dh]
```

新 token 的：

```text
K_new: [B, H, 1, Dh]
V_new: [B, H, 1, Dh]
```

拼接：

```text
K_cache = cat([K_cache, K_new], dim=2)
V_cache = cat([V_cache, V_new], dim=2)
```

## KV Cache 节省了什么

它节省的是重复计算。

不用 KV Cache：

```text
每一步重复算全部历史 token 的 K/V
```

用 KV Cache：

```text
历史 token 的 K/V 只算一次
```

这会显著提升生成速度，尤其是长上下文生成。

## KV Cache 消耗了什么

它消耗显存。

因为每一层都要保存 K 和 V。

大致规模和这些因素相关：

```text
batch size
层数
head 数
head_dim
上下文长度
数据类型
```

上下文越长，KV Cache 越大。

这也是为什么推理系统要做 PagedAttention、prefix caching、量化 KV Cache 等优化。

## Prefill 和 Decode

推理通常分两个阶段：

### Prefill

把 prompt 一次性送进模型，计算所有 prompt token 的 K/V。

```text
输入：完整 prompt
输出：第一批 KV Cache + 第一个生成 token 的概率
```

Prefill 计算量大，但可以并行。

### Decode

每次只生成一个新 token。

```text
输入：上一步生成的 token + KV Cache
输出：下一个 token + 更新后的 KV Cache
```

Decode 每步计算量小，但强依赖前一步，难以完全并行。

## 和训练有什么区别

训练时通常不需要 KV Cache。

因为训练时整段序列已知，可以并行计算所有位置。

KV Cache 主要服务推理，尤其是自回归生成。

## 一句话总结

KV Cache 解决的是：

```text
生成时不要重复计算历史 token 的 K/V
```

它用显存换速度，是大模型推理优化的核心基础。

