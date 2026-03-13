---
title: Diffusion
date: 2025-11-11
tags:
  - genai
---
## 引子

- 将 x 视为你想要滴在水面上的一滴墨水的随机位置 --> 最终遍布整个空间

- 如果我们可以让时间倒流 (比如使用神经网络) --> 我们可以获得整个复杂分布。

Challenges:

1. How to simulate the diffusion process of dataset——the forward process
2. How to inverse this process——the backward process
3. How to train the model——score matching

Denoising diffusion generative modeling is closely related to physics

## Forward Process

> The forward process transforms an image to a noise
>
> **注意**：The forward process is fixed. 没有可学习参数哈！

### 第一步情况

t=1时（依赖于t=0）的分布
$$
q_1(x_1) = sum q(x_0)q_1(x_1|x_0)
$$
考虑 $q_1(x_1|x_0)$，我们的策略是
$$
q_1(x_1|x_0) = N(x_1; sqrt(1-beta_1)*x_0, beta_1I),qquad beta_1 in (0,1)
$$
写成重参数化形式就是 $x_1 = sqrt(1-beta_1) * x_0 + sqrt(beta_1) epsilon_1$

意义：

- 通过 $sqrt(beta_1)$ 向图像添加噪声
- 通过 $sqrt(1-beta_1)$ 减弱/去除 𝑥 中的上下文信息

### 始-终情况

![image-20251120214500945](image-20251120214500945.png)

数学归纳法得（证明不要求）
$$
x_(t+1) = sqrt(prod(1-beta_i))*x_0 + sqrt(1-prod(1-beta_i)) * epsilon
$$
总结一下，就是

- $q_(t+1)(x_(t+1)|x_0) = N(x_(t+1); sqrt(alpha_(t+1))*x_0, (1-alpha_(t+1))I)$
- $x_(t+1) = sqrt(alpha_(t+1))*x_0 + sqrt(1-alpha_(t+1) )epsilon$
- 其中 $alpha_(t+1) = prod(1-beta_i)$

如果我们能设计一个比较好的 $beta_t$ 的策略，使得 $lim_(t->oo) alpha_t ->0$。那么 $q_t(x_t|x_0) -> N(0,I)$ 收敛为一个标准的正态分布。

这个是最经典的，还有各种变种，不过目标是一致的。

## Backward Process

![image-20251120215711433](image-20251120215711433.png)

目标：建模一个有意义的、从后往前的条件分布
$$
p_(t-1)(x_(t-1)|x_t; Theta)
$$

## 模型训练：DDPM

Denoising Diffusion Probabilistic Model (DDPM)

![image-20251120224202159](image-20251120224202159.png)

训练目标：让每一步 p q 都一样

---

损失函数的设计——从最大似然框架出发看看
$$
"minimize " E_(q_0(x_0)) [-log p_0(x_0;theta)]
$$

- 给定任意 $x_0$ ，要计算它由 $p_theta$ 生成的可能性有多大并非易事——不能直接计算概率——隐式模型——求 ELBO

### 分解最大似然函数

![image-20251120220943656](image-20251120220943656.png)

注意：

1. 上图 define $z = x_(1:T)$

2. 第二项就是关于 $q_0(x_0)q_(1:T)(z|x_0)$ 的分布的KL散度，所以非负
   $$
   E_(q_0(x_0)q_(1:T)(z|x_0)) [log( (q_0(x_0)q(z|x_0))/(q_0(x_0)p_0(z|x_0)))]
   $$

> 有的人认为 Diffusion 可以一步步变成层级化的 VAE，和 VAE 太像了，可以类比，trick 也相似。

### 可计算化 ELBO

**推导不是考试内容**，参见 Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020

![image-20251120222239108](image-20251120222239108.png)

注意：

1. 对于任何 q 的相关分布（也就是前向过程的分布），因为前向过程是一个固定的过程，那么就都是可以采样计算的。Samples can be obtained from the forward process，**从而期望可以用均值去逼近，这个变量的期望本身是可计算的。**

2. 问题来到被积函数

   关于 p 的项：这个概率分布已经被神经网络建模了，只要让神经网络的输出和 x0 / xt-1 尽可能的接近，就完成了「最大化概率」/这两项的最小化。单步的东西都可以算，每一步的输入和输出都是确定的。

   ![image-20251120222952322](image-20251120222952322.png)

3. 问题来到中间这个项！==**Why？作业要用！**==

   ![image-20251120223348864](image-20251120223348864.png)

4. 不过，我们因为只算梯度，其实只关心有sigma的项

> [!note]
>
> 高维高斯分布的KL散度
>
> ![image-20251120223427391](image-20251120223427391.png)
>
> 在这里，只关心有sigma的项，其实只有中间 $mu$ 这一项（两个高斯分布均值的某种距离）
>
> ![image-20251120223658155](image-20251120223658155.png)

### 重参数化技巧

预测噪声/预测原始图片，都可以满足去噪目的！

预测一个噪声

DDPM原文代码 Lsimple 算法即如此

### 模型结构

和 Transformer 结构的区别？

Stable Diffusion 使用 transformer 结构，而 Meta 使用 UNet

sinusoidal position embedding ( time embedding ) 时间维度！但其实在 NLP 现在已经不流行了！那么！在 Diffusion 怎么做更好！？还值得研究！

## Connection between DDPM and score function

 ## Understand DDPM through stochastic differential equation

