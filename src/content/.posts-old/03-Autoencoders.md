---
title: Variational Autoencoder
date: 2025-09-19
tags:
  - basic_models
category: Computer
---

# Variational Autoencoder

在深度学习的生成模型领域，变分自编码器（Variational Autoencoder, VAE）是一座重要的里程碑。它不仅继承了传统自编码器的压缩能力，更通过概率论的眼光重新定义了潜在空间（Latent Space），使得我们能够从无到有地生成全新的数据。

---

## Vanilla Autoencoder

自编码器（Autoencoder, AE）本质上是一个“预测自己”的馈送神经网络。它的工作流程非常直观：输入一个信号 $x$，经过编码器 $g$ 压缩成一个低维的向量 $h$，再经过解码器 $f$ 还原回 $x'$ 。

在这个过程中，我们的优化目标是最小化重建误差，即 $||VUx - x||_2$ 。

为了让模型学到有用的特征，我们采用**瓶颈架构（Bottleneck Architecture）**。这意味着中间层变量 $h$ 的维度 $k$ 必须远小于输入维度 $d$ 。如果 $k=d$，模型可能会直接学会一个恒等变换（Identity Matrix），导致模型只是死记硬背而没有起到降维或提取特征的作用 。

然而，朴素 AE 有一个致命的缺点：它的潜在空间是**离散且不连续的**。对于同一个输入，AE 总是映射到一个确定的点 。当你试图从这个空间随机采样一个点交给解码器生成新图像时，由于这些“点”之间充满了未定义的真空地带，生成的往往是毫无意义的噪声。

---

## 从确定性映射到概率分布

为了让自编码器具备生成能力，我们需要对潜在空间进行改造。VAE 的核心动机在于：如果我们能让编码器输出的 $h$ 在训练后服从一个**已知的分布**（如标准正态分布），那么在推理阶段，我们只需要从这个分布中随机采样，就能让解码器生成像样的数据了 。

在 VAE 的架构中，编码器不再直接输出一个向量，而是输出概率分布的参数——通常是**均值 $\mu$ 和标准差 $\sigma$** 。

- **编码过程**：$h \sim g(x)$，即从编码器预测的分布中采样得到 $h$ 。
- **解码过程**：$x' = f(h)$，将采样得到的随机变量还原为数据 。
    

这种改变使得 $h$ 变成了一个随机变量 。在工程实现上，为了处理 $\sigma$ 必须为正值的限制，我们通常让模型预测 $\log(\sigma)$，从而将取值范围从 $(0, +\infty)$ 映射到 $(-\infty, +\infty)$ 。

---

## 解决采样不可导的重参数化技巧

VAE 的训练面临一个严峻的挑战：**采样（Sampling）操作是不可导的**。在反向传播时，梯度无法穿过随机采样过程到达编码器参数 $\theta_{enc}$ 。

为了解决这个问题，我们引入了**重参数化技巧（Reparameterization Trick）**。我们不再直接从 $\mathcal{N}(\mu, \sigma^2)$ 中采样，而是先从标准正态分布中采样一个噪声 $\epsilon \sim \mathcal{N}(0, I)$，然后通过以下公式进行缩放和平移 ：

$$h = \mu + \epsilon \odot \sigma$$

![[image-20250916092140862.png]]

这样一来，随机性被转移到了 $\epsilon$ 这个常数上，而 $\mu$ 和 $\sigma$ 变成了可导的路径，使得整个模型可以进行端到端的梯度下降训练 。

---

## 损失函数的平衡艺术

VAE 的损失函数由两部分组成，共同约束模型的行为：

1. **重建误差（Reconstruction Loss）**：确保解码器能把 $h$ 还原回原始输入 $x$ 。
    
2. **正则化项（Regularization Loss）**：使用 **KL 散度（KL Divergence）** 来衡量编码器输出分布与标准正态分布 $\mathcal{N}(0, I)$ 之间的差异 。
    

$$Loss = ||x - x'||_2 + \lambda KL(\mathcal{N}(\mu, \sigma) || \mathcal{N}(0, I))$$

为什么要加 KL 散度？如果没有这项约束，模型为了降低重建误差，会让 $\sigma$ 趋向于 0，从而退化回死记硬背的朴素 AE 。KL 散度迫使潜在空间变得连续且集中，保证了隐变量空间的连续性和可解释性 。

---

## 概率视角下的 ELBO 推导

从概率论的高级视角来看，VAE 实际上是在最大化数据的**对数似然 $\log P(x)$** 。由于直接计算高维积分不可行，我们转而最大化其**证据下界（ELBO, Evidence Lower Bound）** 。

通过贝叶斯公式和推导，$\log P(x)$ 可以分解为 ：

$$\log P(x) \ge \mathbb{E}_{z \sim q(z|x)}[\log p(x|z)] - KL(q(z|x) || p(z))$$

- **第一项（期望对数似然）**：对应重建准确度，即用隐变量 $z$ 还原 $x$ 的能力 。
    
- **第二项（KL 散度）**：确保后验分布 $q(z|x)$ 接近先验分布 $p(z)$，防止过拟合 。
    

---

## 进阶演化之 VQ-VAE

虽然 VAE 解决了生成问题，但由于它假设潜在空间服从连续的高斯分布，在处理具有明显离散特征的数据（如语音中的音节、图像中的特定纹理）时，往往会产生模糊的生成结果。

**矢量量化自编码器（VQ-VAE）** 针对这一问题提出了改进。不同于 VAE 输出连续的分布参数，VQ-VAE 维护一个**可学习的代码本（Codebook）**。编码器的输出会被映射到代码本中最接近的那个离散向量上。这种离散的潜在表示不仅更符合自然界信息的构成（如语言是由离散单词组成的），还显著提升了生成图像的清晰度和复杂任务的建模能力。

目前的视觉大模型（如某些版本的 Stable Diffusion）底层往往就使用了类似的离散化自编码器技术，将高维像素空间压缩到离散的潜在空间中进行高效生成。


## 0 朴素 AE 架构

**定义**：An autoencoder is a feed-forward neural network whose job is to take **an input x** and **predict x**.
$$
f(g(x)) = x
$$

$$
x'=VUx
$$

**优化目标**：
$$
"minimize" norm(VUx-x)_2
$$

**性质**：

1. f and g can be shallow neural networks
2. Autoencoders are data-specific and learned 和压缩算法不同，且难泛化
3. Autoencoders learn useful properties of data，相比于主成分分析法学习主要成分（见附录）

**注意**：

1. 必须使用瓶颈架构 $(k < < d)$，进行降维操作。否则会收敛于 short-cut solution （如果 $k=d$，可以让 $VU=I$，没有意义）
2. 这个优化目标有闭式解。==（是什么？待补充）==
3. f and g shouldn’t be too complex or powerful，避免学习到「死记硬背」shortcut，也避免过拟合。

## VAE 的动机：让 Autoencoder 变成生成模型

When the encoder is replaced by random noise, the decoder becomes a generative model!

How to make 𝒉 to be a (3)known (1)distribution after (2)training?

## 01 架构：让输入、中间表示变成分布

g 服从一个分布，f 是一个确定性函数，h 是一个随机变量


![[image-20250916084139493.png]]

![[image-20250916084240209.png]]

**注意**：

- The encoder and decoder are not necessarily symmetric
- The encoder and decoder are not necessarily MLP （例如，可以是卷积操作）
- **工程实现**：
  1. 算 log_sigma -- 把 sigma (0,+inf) 映射到 (-inf, inf)
  2. 重参数化技巧（见后）

**潜在空间的性质**

- **AE（自编码器）**: 它的编码器 g(x) 直接将输入 x 映射为一个确定的、单一的潜在向量 h。也就是说，h=g(x)。这个潜在向量是**确定的（deterministic）**。因此，对于同一个输入，AE 总是会得到相同的潜在向量。
- **VAE（变分自编码器）**: 它的编码器 g(x) 不直接输出一个向量，而是输出一个**概率分布**的参数（通常是均值和标准差）。潜在向量 h 是从这个分布中**采样（stochastic）** 出来的。也就是说，h 是一个**随机变量（random variable）**，而不是一个确定的值。

《Masked Autoencoder are scalable vision learner》

## 02 训练：重参数化技巧

> How to make 𝒉 to be a known distribution after ==training==?

**损失函数**（写成与各可训练参数相关的形式）：
$$
cc L = norm(x-x')_2 = norm(x-f_(theta_"decoder")(h))_2, qquad "where "h ～ ccN(g_(theta_"encoder")(x))
$$
其中，$(del cc L)/(del theta_("enc")) = (del cc L)/(del h) (del h)/(del g) (del g)/(del theta_("enc"))$ ，这里面 h 对 g 不可导，采样的过程是不连续的。使用 **重参数化技巧** 解决 sampling 无法求导的问题。

——先采样噪声 $epsilon ～N(0,I)$，再 rescale！

![[image-20250916092140862.png]]

### VAE 实践小结

![[image-20250916092416517.png]]

## 03 让 h 变成已知分布：在损失函数中加入正则化项

> How to make $h$ to be a ==known== distribution after training?

Why we need known distribution? 因为每张图片出来的 $mu$ 和 $sigma$ 都五花八门，真把 encoder 丢掉之后，到底该选怎样的一个分布作为 decoder 的输入？不去对 $mu$ 和 $sigma$ 进行限制（特别是 $sigma -> 0$ 的情况），就会导致模型死记硬背图片（生成很随机？）。

所以，我们需要对 $mu$ 和 $sigma$ 进行限制，「不是那么不一样」，才能做 inference。如何实现呢？

**调整损失函数：L2 范数 -> 加入KL 散度进行正则化**
$$
D_"KL"(q||p) = int q(z) log ((q(z))/(p(z))) dz = bbbE_(z～q(z)) [log q(z)-log p(z)]
$$
![[image-20250916093229665.png]]

## 04 Probabilistic view of VAE

> 为什么冠名 Variational？体现在哪？概率表示？（引入 Variational Lower Bound）
>
> 这是 VAE 的 Paper 原本的讲述方式，设计原则。

### 建模后验分布

**Data generation process**: 

- Assume there is a latent code 𝑧 that guides the generation of 𝑥. （什么是 latent code？比如说，商品的标签。可以建模标签的分布）

**前验分布** $p$：Assume 𝑧 follows a distribution $p(z)$, called the prior distribution. 一般令 $p(z)=ccN(0,I)$

- **后验分布** $q$：利用贝叶斯公式，我们引入后验分布的概念，Assume we have another function/distribution $q_(theta_2)(z|x)$ that can find the region **Variational posterior distribution** 变分后验分布

### 最大似然函数的 ELBO

根据最大似然框架，优化的目标是最大化对数似然函数
$$
P(x) = int p_(theta_1)(x|z)p(z) dz => "maximize "sumlogP(x_i)
$$
这通常在计算上难以处理（涉及具有非线性函数——神经网络的高维积分）

转化为最大化似然函数的**经验下界ELBO**

![[image-20251124104506350.png]]
$$
log P(x) >= bbbE_(z～q(z|x)) [log ((p(x,z))/(q(z|x)))] 
$$
从而，maximize(ELBO) 就可以实现 maximize 对数似然函数。

### ELBO 一个有意义的分解

**经验下界还可以进一步分解为「重建误差」和「正则化项」KL散度**

![[image-20251124104844860.png]]

- **第一项**：用模型推断出的隐变量z，能多大程度上把原始x给还原回来—— 数值越大，说明重建得越准。
- **第二项**：通过约束$q(z|x)$（编码器的隐变量分布）接近先验 $p(z)$（通常假设为标准正态分布），避免模型过拟合，同时保证隐变量空间的连续性和可解释性。

## 05 VQVAE

==【待补充】==

