# Transformer

Transformer 本质上是一种处理序列的神经网络结构。

它最早在论文 **Attention Is All You Need** 里被提出，主要用于机器翻译。后来你看到的很多大模型，比如 GPT、BERT、ViT，也都和 Transformer 有很深的关系。

先放经典架构图：

![Transformer经典架构图|367](图/Pasted%20image%2020260508093635.png)

## 它想解决什么问题

在 Transformer 之前，处理句子常用 RNN、LSTM。
它们的思路比较自然：从左到右，一个词一个词读。
```text
我 -> 喜欢 -> 吃 -> 苹果
```

但这种方式有两个问题：
1. **慢**：后一个词必须等前一个词处理完，很难并行。
2. **远距离信息容易丢**：句子很长时，前面的信息传到后面会变弱。

比如：
```text
我昨天在超市买的那个红红的、很甜的苹果很好吃。
```
“好吃”真正评价的是“苹果”，但中间隔了很多词。模型如果只按顺序慢慢传，很容易忘掉前面的核心词。
Transformer 的想法是：**不要一个词一个词排队读，而是让每个词直接去看整句话里和自己有关的词。**
这就是 Attention。

## 学 Transformer 前建议先补的前置知识

这是一个很庞大的体系，如果之前没有系统学过深度学习，直接看架构图很容易被一堆模块名劝退。

所以我建议先按这个顺序看：

1. [损失函数](前置知识/损失函数.md)：知道模型为什么要训练。
2. [梯度消失](前置知识/梯度消失.md) 和 [梯度爆炸](前置知识/梯度爆炸.md)：知道深层网络为什么不好训。
3. [激活函数](前置知识/激活函数.md)：知道神经网络为什么需要非线性。
4. [词嵌入](前置知识/词嵌入.md)：知道文字怎么变成向量。
5. [softmax](前置知识/softmax.md)：知道分数怎么变概率。
6. [归一化](前置知识/归一化.md)、[BatchNorm](前置知识/BatchNorm.md)、[LayerNorm](前置知识/LayerNorm.md)：知道为什么要稳定数据分布。
7. [Add](前置知识/Add.md)：知道残差连接为什么重要。
8. [正则化](前置知识/正则化.md) 和 [DropOut](前置知识/DropOut.md)：知道怎么缓解过拟合。

前置知识不用一次全背下来。你只要能先理解每个东西“解决什么问题”，后面看 Transformer 会顺很多。

## Transformer 的主线

Transformer 可以拆成四个关键问题：
### 1. 文字怎么进入模型

模型不能直接读中文、英文，它只能处理数字。
所以第一步是把每个词变成向量，也就是词嵌入。
但词嵌入只告诉模型“这个词是什么意思”，没有告诉模型“这个词在第几个位置”。
所以 Transformer 的输入是：
```text
输入向量 = 词嵌入 + 位置编码
```
对应笔记：
- [词嵌入](前置知识/词嵌入.md)
- [位置编码](关键模块/位置编码.md)

### 2. 每个词怎么看其他词

这是 Transformer 最核心的部分：[[注意力机制]]
当模型处理某个词时，它会计算这个词和其他词的相关性，然后决定应该重点吸收谁的信息。
比如：
```text
我 喜欢 吃 苹果
```
处理“吃”时，模型可能会更多关注“苹果”。
这一步靠的是 Q、K、V：
```text
Q：我想找什么
K：每个词能提供什么线索
V：每个词真正携带的信息
```
对应笔记：[注意力机制](关键模块/注意力机制.md)
### 3. 为什么要多头
一句话里的关系不止一种，一个注意力头可能关注“动作和对象”：
```text
吃 -> 苹果
```
另一个注意力头可能关注“谁在做动作”：
```text
喜欢 -> 我
```
所以 Multi-Head Attention 可以理解成：**让模型从多个角度同时理解句子。**
### 4. 编码器和解码器怎么配合
原始 Transformer 是编码器-解码器结构。
如果做机器翻译：

```text
输入：我 喜欢 猫
输出：I like cats
```

编码器负责读懂中文，解码器负责生成英文。
编码器像读题，解码器像答题。

对应笔记：[编码器与解码器](关键模块/编码器与解码器.md)

## 一层 Transformer 在做什么

如果只看编码器的一层，它大概是：

```text
输入
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Feed Forward
  ↓
Add & Norm
```

可以这样理解：

```text
注意力层：让词和词交换信息
Add：把原来的信息保留下来
LayerNorm：把数值分布拉稳
Feed Forward：每个位置自己再加工一下
```

很多层叠在一起后，模型就能从浅层的词面关系，逐渐学到更复杂的语义关系。

## Encoder、Decoder、GPT、BERT 的关系

原始 Transformer 有 Encoder 和 Decoder。
后来很多模型会只取其中一部分：

| 模型类型 | 用什么结构 | 适合做什么 |
| --- | --- | --- |
| Encoder-only | 只用编码器 | 文本理解、分类、检索，比如 BERT |
| Decoder-only | 只用解码器 | 文本生成，比如 GPT |
| Encoder-Decoder | 编码器 + 解码器 | 翻译、摘要、输入输出都较复杂的任务 |

所以你学原始 Transformer，不只是学一个老模型，而是在学很多现代模型的共同底座。

## 图片如何下载后正常加载

这个文件夹里的图片统一放在：
```text
AI/transformer/图/
```
笔记里统一使用 Markdown 相对路径，例如：
```markdown
![Transformer经典架构图](图/Pasted%20image%2020260508093635.png)
```
这样别人从 GitHub 下载整个仓库后，只要不改目录结构，图片就能正常加载。

这里不要使用本机绝对路径，比如：
```text
D:/obsdian/note/AI/transformer/图/xxx.png
```
因为别人电脑上没有这个路径。
也尽量少用 Obsidian 的图片语法，也就是这种双中括号写法：
```markdown
! [ [ Pasted image.png ] ]
```
这种在 Obsidian 里方便，但 GitHub 网页和普通 Markdown 预览器不一定认识。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
具体[[AI/transformer/代码实现|代码实现]]点这
