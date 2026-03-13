---
title: Transformer
date: 2025-09-25
tags:
  - basic_models
category: Computer
---


---

![[image-20250926114833272.png]]

Content

- Transformer components
  - Attention
  - FFN
  - Positional Encoding 老师推荐可以尝试研究，也有很多国人的成果

- Transformer variants

期中会考的！！

## Attention

- Assume each word is represented as a vector $x_i in RR^d$ (word meanings) **(aka word embedding)**
  $$
  [x_1, x_2, ..., x_n] -> [z_1, z_2, ..., z_n], z_i in RR^d
  $$

​	==Attention is used to **obtain context-dependent word meanings**==

- We hope $z_i$ contains more **surrounding information (so-called context)** than $x_i$.
- The information to be contained should be
  - Selective 精准的
  - Word-dependent 依赖的

### 理解 Attention 机制：Attention 是软化的哈希表

- Hash table - A hash table consists of **(key, value) pairs**

  1. A query $q$ arrives, we first **search** all keys and find out the key $k**$ that exactly matches $q$.

  1. Return the corresponding $v**$ 

     If there is no exact match, return empty

- **Attention**：A **soft version** of matching a hash table

  Attention mechanism:
  1. A query $q$ arrives, we **calculate the correlations** between $q$ and all keys
  2. **Linearly combine** values according to the correlation 
  3. Return the **linear combination**

![[image-20250926104819672.png]]
### Attention 的计算

算 q 与 k 的值，**Score** 要做一次 normalization

$"Score" = (q_1*k_1)/sqrt(d_k)$ ：为什么要除？防止模型陷入「只看某个词」，不然容易在 softmax 出现有的几乎为1，其他几乎为0的情况。缓解初始时 softmax 出现极值。To avoid model converge to bad local minima at the beginning of the training **==【如何理解 待补充】==**

#### The compact form of attention

parallelizable matrix computations

$Q,K,V in RR^(n xx d_k)$

#### Multi-head attention

- 每个 head 都有自己的一组 Wq, Wk, Wv
- 为什么要 multi-head？增加参数量
- 如何 output = concat[AV]？sum / num_head

## Why FFN

- Strictly speaking, attention is a non-linear transformation, 毕竟有一个 softmax 操作 
  $$
  "softmax"((QK^T)/sqrt{d_k})V
  $$

- 但它的功能和线性的差别没那么大，因为 Attention only collects and organizes information, **without processing information**

- 所以引入 FFN
  $$
  "FFN"(z_i) = W^2 sigma (W^1 x_i)
  $$

- 和经典的区别：不使用 ReLU，而是 GeLU，希望在0附近有导数，在0附近增加一些信息【为什么】

- The dimension of the hidden layer $W^1$ is usually large，比较有效

## Add & Norm

### Residual connection

Residual connections are also thought to smooth the loss landscape and **make training easier**! 经验表明，有助于梯度下降优化。

![[image-20250926112212385.png]]

![[image-20250926112204425.png]]

### Layer normalization

历史上有很多很多的 Normaliztaion ... CV 常用 Batch Normalizaion 2D，其他还有 InstanceNorm2D（早期 Image gen model), GroupNorm. 他们官方的介绍都是
$$
y = (x-"E"[x])/sqrt("Var"[x]+epsilon) * gamma + beta
$$

- 甚至，You may find paper-code mismatch。哈哈哈
- Easy to write **<u>code bugs</u>**
- 如何区分它们？
- Norm 扮演了什么角色？目前没有一个完整的答案。

在 NLP 中，

- Tensor shape: 

  `batch_size(8) * sequence_length(256) * embed_dim(512)`

- `T[i][j][k]`: the 𝑘-th element in the embedding of the 𝑗-th token in the 𝑖-th sentence

- `T[i][j][:]`: the embedding vector of the 𝑗-th token in the 𝑖-th sentence

- `T[:][j][k]`: the 𝑘-th elements in the embedding of the 𝑗-th token in all the sentence

**Different normalization** methods calculate mean and variance along **different dimensions**

- **Layer normalization**: normalize along the embedding dimension ==**【必考】**==
  - For each (sentence, position) in the batch
    - Obtain the (intermediate) embedding
    - Calculate its mean and variance
    - Normalize the embedding
  - End For

## Post-LN / Pre-LN Transformer

![[image-20250926113814194.png]]

- 左图没有直接线性通路，是全非线性的；右图则有直接线性通路。
- 左图最大的问题：**优化**非常不稳定，对超参和学习率非常敏感，学习率必须要经过设计。Warm-up。

![[image-20250926114139742.png]]

![[image-20250926114703353.png]]

## Positional Encoding

- **Attention & FFN modules** don’t consider **positional information**
- 相隔越近的token应该自注意力分数更高！

### Absolute positional encoding

![[image-20250926115356254.png]]

### Relative positional encoding

- **加性相对位置矩阵**

![[image-20251025161041918.png]]

![[image-20251025161050719.png]]
  - 因为注意力分数算出来是一个n×n的矩阵，其中横纵坐标就是这个词与另一个词。所以加的这个位置编码 $B$ 矩阵。也是一个「对角」矩阵，对于相对位置一致（顺序相同且间隔词数相同）的一对词。相对距离 $b_(ij) = f(i-j)$ 相同.

- **旋转位置编码 Rotational positional encoding**：乘以旋转矩阵。Rotational position encoding method, encoding the relative distance between every two positions through rotation matrix (全部国人做的)

  - 把相对位置信息建模成一个小旋转矩阵 $bb R_(ij) = f(i-j)$，可以把这个旋转矩阵嵌入到注意力分数计算中 $Q$ 和 $K$ 乘积的中间。

![[image-20251025161636026.png]]



