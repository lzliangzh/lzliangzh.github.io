---
title: Generative Adversarial Networks
date: 2025-10-24
tags:
  - genai
---




[TOC]

## From complex distribution to simple distribution

从简单分布出发，把简单分布变换为复杂分布

Random variable $X$ follows a complex distribution $Q$, how can we generate data from $Q$?

---

假设存在且已知 $Q$ 的累积概率密度函数 $F(x)$ ，并可以计算出其反函数 $F^(-1)(x)$.

> [!note] 
>
> 累计概率密度函数 $F$ 的性质：
>
> - $F$ 是单调不减的；
>
> - $lim_(x->oo)F(x)=1$; $lim_(x->-oo)F(x)=0$
> - $F$ 是右连续的。

Algorithm

- Sample $Z$ from a **uniform** distribution
- **结论**：$X=F^(-1)(Z) ～ QQ$

$$
P(X<t) = P(F^(-1)(Z)<t) = P(Z<F(t)) = F(t)
$$

- Samples from any complex distribution can be generated from simple distributions
- **如何建模 $Q$ 呢**

![image-20251017102639942](image-20251017102639942.png)

## How to estimate the difference between generated data $x_theta$ and real data $x_star$?

- 如何设计一个 metrics，衡量模型生成数据的概率分布 $Q$ 与Ground Truth 分布 $P$ 的差异？
  - 单点差异 -> 「距离」 (e.g. 欧氏距离)
  - 两个点集的差异 -> 平均距离；最小距离…重合了不好。
- 衡量两个分布的差异：**F-divergence; Integral probability metric (IPM); Wasserstein distance**
- 不同 metrics 导出不同的模型

## F-divergence

> 无法保证对称性，所以叫做 divergence。如果具备互换性，就是 distance

- 已知：分布 $P(x)$（概率密度函数 $p(x)$），$Q(x)$ 
- 设计一个 Function $f: [0, +oo] -> [-oo,+oo]$
  - Convex; Non-negative; $f(1) = 0$; finite

$$
D_(f) (P || Q) = int f((dP)/(dQ))dQ = int f((p(x))/(q(x)))q(x) dx
$$

- 为什么合理？$p(x)=q(x)$ 时，$int f(..) = intf(1) = int0$

**性质**：

- $p(x) = q(x) => D=0$

- 非负性（证明使用微积分柯西不等式）
  $$
  D_(f)(P||Q) = int f((p(x))/(q(x)))q(x)dx >= f(int((p(x))/(q(x)))q(x)dx)
  $$

  $$
  = f(int p(x) dx) = f(1) = 0
  $$

- 线性性，凸性

**F-divergence 和其他散度的转化**：

- 设计 $f(t) = tlogt$，此时变成 **KL-divergence**

$$
D_(f)(P||Q) = int (p(x))/(q(x)) log ((p(x))/(q(x))) q(x) dx = int p(x) log ((p(x))/(q(x))) dx
$$

  - 设计 $f(t)=1/2 |t-1|$，此时变成 **total variation distance** 全变差距离

    ![image-20251017105307158](image-20251017105307158.png)

  - 设计 $f(t) = -(t+1) log ((t+1)/2) + tlogt$，此时变成 **Jensen-Shannon(JS) divergence**

    ![image-20251017105349377](image-20251017105349377.png)

##  GAN

- **概率论视角**：优化 JS 散度
- **深度学习老奶奶能听得懂视角**：先训练 D，判别是真是假；再训练 G，生成能骗过判别器的图片

### minimize JS-divergence

- $P -> P_star$：**data** distribution
- $Q -> P_theta$：**model (generated)** distribution

![image-20251017111145652](image-20251017111145652.png)

![image-20251017111828174](image-20251017111828174.png)

- ==能不能**估计或者学习** $P_star, P_theta$？==

- 注意到
  $$
  (P_star(x))/(P_star(x)+P_theta(x)) + (P_theta(x))/(P_star(x)+P_theta(x)) = 1
  $$

![image-20251017112605717](image-20251017112605717.png)

- Build a binary classifier FFN $D_(theta')(x)$ to identify which distribution is x likely coming from

$$
D_(theta')(x) ~~ (P_star(x))/(P_star(x)+P_theta(x))
$$

希望用左边代替计算右边的值，估计之——类比为二分类任务。训练任务就转化为
$$
underline["Minimize" qquad E_(x～P_star)[log D_(theta')(x)] + E_(x～P_theta)[log(1-D_(theta')(x))]]
$$

### Adversarial Training of  $D_(theta'), G_theta(z)$

**注意**：下面 $x～P_theta$ 又可以写作 $G_theta(z),z～ccN(0,I)$ 这样一个「生成模型」的形式

![image-20251017114048106](image-20251017114048106.png)

![image-20251017113622658](image-20251017113622658.png)

![image-20251017113638094](image-20251017113638094.png)

### Overview

... (d) 最终

![image-20251017115447589](image-20251017115447589.png)

### Algorithm

![image-20251017115056550](image-20251017115056550.png)

### Tough training of GAN

![image-20251024102245573](image-20251024102245573.png)

- 什么是「**mode collapse**」？训练数据集中成簇，导致GAN只拟合一个簇也可以loss很低，只会做一种模式的任务。
- 2 Reading material: 
  - Generative Adversarial Nets
  - Large Scale GAN Training for High Fidelity Natural Image Synthesis.


## IPM -> Wasserstein Distance (aka Earth Mover's Distance)

Integral probability metric
$$
| E_(X～P)[f(X)] - E_(X～Q)[f(X)] |
$$
缺点：与 $f$ 的选择非常相关。改进为：
$$
D_F(P,Q) = "sup"_(f in F)  | E_(X～P)[f(X)] - E_(X～Q)[f(X)] |
$$
**Wasserstein distance** $W(P,Q)$：where $F$ is the **set** of all 1-Lipchitz **function** (利普西斯系数为1，即导数为1)。

---

**瓦萨斯坦距离的等价形式**：
$$
W(P,Q) = "inf"_(gamma in Gamma(P,Q)) E_((X,X')～gamma)|X-X'|
$$
where $Gamma(P,Q)$ is a set of distributions defined on $X xx X$. Each $gamma ∈ Gamma$ satisfies
$$
int gamma(x,y)dy = p(x), qquad int gamma(x,y)dx = q(y)
$$
**如何理解瓦萨斯坦距离的意义**：愚公移山 (aka Earth Mover's Distance)

![image-20251024104411222](image-20251024104411222.png)

- Assume the above (mass) distribution is 𝑝(𝑥)

- Assume you’d like to transform 𝑝(𝑥) to 𝑞(𝑥)

- $gamma(x,y)$ is **the amount of mass** that you would like to move from location 𝑥 to location 𝑦

  $p(x),q(y)$ 发出和接受（目标）的土的数量。是一个「转移矩阵」，标识了每一个点应该如何变换，能把 $p$ 搬运成 $q$.
  $$
  int gamma(x,y)dy = p(x), qquad int gamma(x,y)dx = q(y)
  $$
  有很多种的选择，要找出里面成本最小的一种.

- Assume that there is a cost function if you move a unit from 𝑥 to 𝑦
  $$
  "cost"(x,y) = |x-y|
  $$

  $$
  "inf"_(gamma in Gamma(P,Q)) int "cost"(x,y) gamma(x,y) dxdy = "inf"_(gamma in Gamma(P,Q)) E_((X,X')～gamma)|X-X'|
  $$

> [!note]
>
> More concept if you want to learn
>
> - Optimal Transport
> - Sinkhorn algorithm
> - Kantorovich-Rubinstein duality

## WGAN

Learn generative model with (minimizing) Wasserstein distance
$$
"minimize " W(P_star,G_theta(z))
$$
![image-20251024112259826](image-20251024112259826.png)

How to design **Lipchitz-bounded** neural networks? 

1. Weight clipping: continuous function in bounded closed domain has finite Lipchitz constant 有界闭域上的连续函数具有有限的利普希茨常数
2. 但是有限的利普希茨常数≠小的利普希茨常数。一个可微函数是 1 - 利普希茨函数，当且仅当它在每一处的梯度范数至多为 1。所以一种想法是，**直接限制 $f_(theta')$ 输出相对于其输入的梯度范数**

上述 Loss 添加一个正则化项：
$$
-lambda bbbE_(X～hat(P)) [(norm(grad_Xf_(theta')(X))_2-1)^2]
$$
使用负号，这样训练 $f_(theta')$ （即调整 $theta'$ 参数）时，「最大化」的训练目标就会最小化这一项，即让 $grad_X f_(theta')(X) = (del f)/(delX)$ 收敛于1.

## GAN’s Position: Explicit vs. Implicit

- 显式模型：知道模型参数；给定任何数据（真实vs生成）可以计算这个数据出现的概率；使用最大似然方法训练。如 AR。
- 隐式模型：知道模型参数；给定任何数据（真实vs生成）**不能**可以计算这个数据出现的概率（如 VAE 中的 ELBO 才可以计算），**不能**使用最大似然方法训练。
- **有没有一个大一统的模型？**

![image-20251017103132025](image-20251017103132025.png)

