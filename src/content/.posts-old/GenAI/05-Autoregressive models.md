---
title: Autoregressive
date: 2025-10-10
tags:
  - genai
---


[TOC]

- **自然语言处理的生成模型方法非常成熟，已经收敛了。**
- 统计-时序分析，也有个自回归模型

- 描述 训练 使用自回归模型

对数据集进行概率建模 --> 高维空间的概率模型 拆解为每一维联合概率

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20251010102510585](image-20251010102510585.png) | ![image-20251010102615778](image-20251010102615778.png) |

## 自回归范式 General setting of Autoregressive model 

- 我们处理的数据是高维数据 (向量)： $"Data "x = (x_1,x_2,...,x_d)$
- 目标：建模 $P(X=x)<=> P(X_1=x_1,...,X_d=x_d)$
  - **联合概率拆解为条件概率** (链式法则) $=P(X_1=x_1)P(X_2=x_2,...,X_d=x_d|X_1=x_1)$

![image-20251010103224985](image-20251010103224985.png)

- **自回归模型是什么** —— If I have a parametric model with parameter 𝜃 which can estimate the following thing
  $$
  P_theta (X_i=x_i | X_1=x_1,...,X_(i-1)=x_(i-1))
  $$
  (Left-to-right factorization (standard in NLP), also Right-to-left factorization)

  > factorization = in which (*dimension*) *order* you generate data
  >
  > - 对于图像，如何选择生成的顺序？
  > - 并不是图片生成的主流模型
  >
  > ![image-20251010104258988](image-20251010104258988.png)

## 自回归模型

- [Method] How to make it universally work for any $i$
- [Optimization] How to train it
- [Evaluation] How to compare two models
- [Inference] How to use it

## Modelling the language: LM

### N-gram LM

$theta$ Directly estimated from data. 直接数数。

n-gram: $n$ adjacent 邻接的 letters. Unigram Bigram Trigram

- **Markov assumption**: assume $x_i$ doesn’t depend on ***far away*** tokens 固定的比较短的长度
  $$
  P(X_i=x_i | X_(i-k)=x_(i-k), ..., X_(i-1)=x_(i-1))
  $$
  
- 转化为条件概率
  ![image-20251010105715993](image-20251010105715993.png)
  
  ![image-20251010105959145](image-20251010105959145.png)

- Disadvantages of n-gram LM：

![image-20251010110255486](image-20251010110255486.png)

### Fixed-window neural LM

不再使用直接统计
$$
P(X_i|X_(i-k)=x_(i-k), ... , X_(i-1)=x_(i-1)) = "NN"(x_(i-k),...,x_(i-1); theta)
$$
![image-20251010110706945](image-20251010110706945.png)

- **Advantages**: 1) No **memory** issue; 2) Parameters = embeddings + hidden layer; 3) No **sparsity(稀疏)** issue.
- **Disadvantages**: 1) **Prefixed** window size $N$; 2) Not enough **ability** for language understanding.

### RNN LM

- 去除定长上下文 $n$ 的限制 --- 回到对 Autoregressive 的定义
- 谷歌翻译的底层技术

![image-20251011162053436](image-20251011162053436.png)

![image-20251011162249077](image-20251011162249077.png)

- 区别：1）定长与非定长；2）非线性的次数更多，能够理解更复杂的信息。

简洁表达：
$$
h_t = f_W(h_(t-1), x_t)
$$

- $h_i$ : 每一状态「历史信息」，$h_(t-1)$ is previous state
- $x_t$: Current input.
- $f_W$: Some function with parameter $W$.

## Training

### RNN Training

- **数据集**：Given a dataset 𝐷 consisting of a set of sentences : 𝐷 = {𝑠1, 𝑠2, ... , 𝑠𝑁}

- **最大似然估计**：We aim to maximize the likelihood of $P_theta (forall i, S_i=s_i)$

  (We hope to have 𝜃 that can sample out dataset 𝐷 with higher probability)

- **最小化负对数似然函数**：

  因为句子 $S$ 是独立同分布采样出来的，独立性可以把联合分布拆解为乘积

  $$
  min [-log P_theta (S_1=s_1, ..., S_N=s_N) <=> sum -log P_theta (s_i)]
  $$
  对于句子 $S_i$ （句长 $n$）中的每一个词元 $X_i$，
  $$
  -log P_theta (s_i) = - sum_(j=1)^n log P_theta (x_j | x_1,...,x_(j-1))
  $$
  
- 因此，Given a (batch of) sentence:

  - Calculate the loss terms one by one through time
  - Backpropagate the gradient of parameter 𝜃 through time (BPTT)

> [!note]
>
> **这种 Training 方法也被叫做 Teacher forcing：默认前面出现的所有 Token 都是 Ground Truth，强制模型预测下一个词。**

Pros:

- Can be applied to any-length sentence
- Fixed number of parameters
- Highly non-linear （每经过一次 RNN 都有一次 Non-linear）

 Cons:

- Extremely slow training (linearly depends on sentence length) loss 的计算只能串行，不能并行，必须先看到前面的词，才能计算 Hidden State。
- Optimization problems (check papers related to LSTM) 梯度消失与梯度爆炸
- **现代 LLM 是如何摆脱这些 Cons 的？对 Transformer 深入探究**

### Modern LLM Training

#### Attention Mask

> How to make attention mechanism an AR? -- Attention Mask

- self-attention 的计算可以看到所有输入的信息，我们需要强行让 self-attention 只看到前面的 Token 的信息，以符合 AR 的输入与输出处理。(The attention can only be calculated on previous tokens)
- **Attention Mask**：也就是说，加权 softmax 时，只能在前面的 Token 上计算（0），其他要被置为 $- oo$。由此设计一个上三角矩阵。--> 要让模型看到的位置保持不变（+0），不想让模型看到的位置变成负无穷 --> 经过 softmax 变化后，负无穷位置的 normalize 的结果为 0 --> 保证了 Transformer 只看到前面的信息，不看到后面的信息。

![image-20251017093948927](image-20251017093948927.png)

- The compact form of decoding only attention：都可以并行。FFN 的存在也使得 Tf 有很多非线性。

![image-20251017094017886](image-20251017094017886.png)

![image-20251017094337074](image-20251017094337074.png)

#### LLM Training Tips

1. Mixed precision training (混合精度训练)

   - 为什么混合？对于一些步骤（如除法）采用高精度，对于矩阵乘法（对精度要求低）采用低精度。Automatic Mixed Precided Pytorch混合精度训练范式，可以加速两倍

   - fp32, fp16 (可能会经常出现除0的问题) and bf16 (>=A100, extremely hard to train LLM with V100，卡好常用)

   - Different components use different precisions

   - Check


2. Large-batch training (Production-level model)

   - mini-batch 经常加噪音

   - Training data (3.4B~4T Token) 否则模型收敛慢

   - 1M tokens per batch at least (512 token/sen * 2048 sen)

   - Gradient accumulation 显存有限怎么办？多batch直到觉得模型见过足够多的数据后（mini-batch模拟large-batch），再更新模型参数。累积历次梯度，最后取平均

```python
device ="cuda"
model.to(device)
gradient_accumulation_steps=2
for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    ###
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if(index+1)%gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad() # 梯度清零
    ###
```

3. Ways to stabilize training (Production-level model)

   - Small dropout

   - Learning rate warm-up

   - Gradient clipping

## Evaluation

### Perplexity 

特别像 ~ Negative log likelihood，但 nLL 对长度敏感（句子越长 nLL 越大）
$$
"nLL"(s;theta) = -log P_theta (s)
$$
Per-word negative log-likelihood (average):
$$
"WnLL" (s, theta) = -1/n log P_theta (s)
$$
Perplexity: 
$$
"ppl"(s;theta) = exp("WnLL"(s;theta))
$$
Other metric (generation): BLEU (often used on 'retrival'), ROUGE...(learn from NLP courses)

![image-20251017143408775](image-20251017143408775.png)

### 下游典型任务

## Inference

- 推理策略：stochastic sampling（纯随机）; 计算出分布，sample 出 token；可能带来过多的随机性。GPT (如下)采用的是 top-k sample，选出前k个概率最大的再随机采样。
- 此外还有 top-p sample（选择概率>p的作为候选集）。
- 极端的情况 top-1 sample 就是 greedy search

![image-20251017143758130](image-20251017143758130.png)

![image-20251017144049158](image-20251017144049158.png)



## Cost

> RNN LM v.s. Transformer LM
>
> - Which one **trains** faster, why?
> - Which one **generates** sentence faster, why?

### ![image-20251017193808498](image-20251017193808498.png)

推理： RNN O(n)，Transformer O(n^2)（要和前面所有信息算 Attention，每次计算越来越多）

![image-20251017194015418](image-20251017194015418.png)



### Computational cost of attention



### Inference cost



### Training cost 

