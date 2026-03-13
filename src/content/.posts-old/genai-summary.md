---
date: 2025-11-25
tags:
  - genai
title: "Summary: Mathematical principles of generative models"
---

## VAE

### 最大似然函数的 ELBO

根据最大似然框架，优化的目标是最大化对数似然函数
$$
P(x) = int p_(theta_1)(x|z)p(z) dz => "maximize "sumlogP(x_i)
$$
这通常在计算上难以处理（涉及具有非线性函数——神经网络的高维积分）

转化为最大化似然函数的**经验下界ELBO**

![image-20251124104506350](image-20251124104506350.png)
$$
log P(x) >= bbbE_(z～q(z|x)) [log ((p(x,z))/(q(z|x)))] 
$$
从而，maximize(ELBO) 就可以实现 maximize 对数似然函数。

### ELBO 一个有意义的分解

**经验下界还可以进一步分解为「重建误差」和「正则化项」KL散度**

![image-20251124104844860](image-20251124104844860.png)

- **第一项**：用模型推断出的隐变量z，能多大程度上把原始x给还原回来—— 数值越大，说明重建得越准。
- **第二项**：通过约束$q(z|x)$（编码器的隐变量分布）接近先验 $p(z)$（通常假设为标准正态分布），避免模型过拟合，同时保证隐变量空间的连续性和可解释性。

## GAN

### 数学基础：散度——测量两个分布的差别

$$
D_(f) (P || Q) = int f((dP)/(dQ))dQ = int f((p(x))/(q(x)))q(x) dx
$$

性质：值域、非负性（用微积分柯西不等式证明）、线性性、凸性

和其他量的转化：

| f的选取                              | 名称                             | 公式       |
| --------------------------------- | ------------------------------ | -------- |
| $f(t) = tlogt$                    | KL散度                           | 易，略      |
| $f(t) = -(t+1)log((t+1)/2)+tlogt$ | JS散度                           | 会证明了，没问题 |
| $f(t) = 1/2\|t-1\|$               | 全变差距离 total variation distance |          |


### 训练框架：优化JS散度

GAN的训练目标是，尽量使得model (generated) distribution $P_theta$ 接近于data distribution $P_star$ ，其中接近程度的衡量即是使用JS散度量化。故而训练目标为
$$
"minimize " KL(P_star || (P_star+P_theta)/2) + KL(P_theta || (P_star+P_theta)/2)
$$
代入 KL 散度公式并消去系数
$$
"minimize " bbbE_(P_star) log [(P_star(x))/(P_star(x)+P_theta(x))] + bbbE_(P_theta) log [(P_theta(x))/(P_star(x)+P_theta(x))]
$$
注意到里边两项加起来为1，引入判别器 $D_(theta')(x)$ 建模 $x$ 来自数据真实分布的概率 
$$
D_(theta')(x) ~~ (P_star(x))/(P_star(x)+P_theta(x))
$$
为了让损失函数能够对模型参数 $theta$ 进行优化，代入 $G_theta(z)～P_theta$。损失函数的最终形式为

![image-20251017113638094](image-20251017113638094.png)

> [!warning]
>
> **注意**：maximize操作，用梯度上升；minimize才是梯度下降。

## WGAN

### 数学基础：Wasserstein 距离及其意义

$$
W(P,Q) = "inf"_(gamma in Gamma(P,Q)) E_((X,X')～gamma)|X-X'|
$$

$Gamma(P,Q)$ 是一个由所有定义在 $X xx X$ 上的联合概率分布 $gamma$ 组成的集合，满足
$$
forall gamma in Gamma,qquad int gamma(x,y)dy = p(x), qquad intgamma(x,y)dx = q(y)
$$
![image-20251124231718887](image-20251124231718887.png)

### 训练框架：优化瓦萨斯坦距离

![image-20251124231754673](image-20251124231754673.png)

上述 Loss 添加一个正则化项：
$$
-lambda bbbE_(X～hat(P)) [(norm(grad_Xf_(theta')(X))_2-1)^2]
$$
使用负号，这样训练 $f_(theta')$ （即调整 $theta'$ 参数）时，「最大化」的训练目标就会最小化这一项，即让 $grad_X f_(theta')(X) = (del f)/(delX)$ 收敛于1.

## Flow

### 数学基础：Change of Variables Theorem

随机变量的变量替换定理，实现了两个PDF函数之间的转化（Function Mapping + Rescale Volume）。

**一维情形**：设 $z～ccN(0,1), x = g(z)$。$g$ 可微可逆，$x,z$ 的PDF分别为 $p_X(x),p_Z(z)$。根据CDF的等价关系，两边对 x 求导并可选地进行变量替换，有
$$
p_X(x) = p_Z(z) [g^(-1)(x)]'=p_Z(z)[g'(z)]^(-1)
$$
**高维情形**：$z～ccN(0,I)$，$g$ 微分同胚 Diffeomorphisms（高维可微可逆）
$$
p_X(x) = p_Z(z)|det D_(g^(-1))(x)|
$$

> 与VAE的联系、参数化技巧

### 模型结构

关键难点在于设计可以方便地（1）计算、（2）求逆、（3）计算雅可比矩阵行列式的函数类 $gin G$。这个函数类的满足两条性质：

1. 复合可逆：可逆函数的复合仍然可逆
2. 复合可微：分解后的函数及其逆都可微

$$
p_X(x) =  p_Z(z)|det D_(g^(-1))(x)| =  p_Z(z) |det D_(g_N^(-1))(x)| ... |det D_(g_1^(-1))(x)|
$$

把复杂网络结构拆解为基本 Layer 的设计，如何构建最基本的单元？

- 逐元素流

![image-20251125073940331](image-20251125073940331.png)

- 线性流

![image-20251125073934044](image-20251125073934044.png)

$g(z) = Az+b$

$D_(g)(z) = A, D_(g^(-1))(x) = A^(-1)$

$det D_(g) = |A| = proddiag(A)$

- Computing the determinant for general matrix requires $𝑂(d^3)$ cost
- Computing the inverse requires $𝑂(d^2)$ cost

- 残差流

![image-20251125074240857](image-20251125074240857.png)

![image-20251125074252245](image-20251125074252245.png)

### 训练

应用变量替换公式，能够直接在最大似然函数中显式计算出真实data的生成概率并最大化它。所以通过MLE训练，其中待训练的参数在 $g$ 上。
$$
ccL = -log p_X(x) = -(log p_Z(g_theta^(-1)(x)) + log|det D_(g_theta^(-1))(x)|)
$$
至于推理，只需遵循模型结构，先从先验分布中采样一个潜变量
$$
z～p_Z(z) quad("usually"=ccN(0,I))
$$
再用前向网络把潜变量映射到数据分布中 $x = g(z)$ 即可

## Diffusion

### 模型结构

逐步加噪模型建模为马尔科夫链，前向分布 $q(x_t) = q(x_0) prodq(x_k|x_(k-1))$（真实分布）

前向过程（单步）
$$
q(x_t|x_(t-1)) = ccN(sqrt(alpha_t)x_(t-1), (1-alpha_t)I)
$$
前向过程（一步到位）
$$
q(x_t|x_0) = ccN(sqrt(bar(alpha)_t)x_0,(1-bar(alpha)_t)I)
$$
重参数化（以前向过程一步到位为例）
$$
x_t = sqrt(bar(alpha)_t)x_0 + sqrt(1-bar(alpha)_t) epsi quad "where " epsi ～ccN(0,I)
$$
后向过程，引入对后向分布的参数化建模——建模一个有意义的、从后往前的条件分布 $p_theta(x_(t-1)|x_t)$，并对其进行训练，使得
$$
p_theta (x_(t-1)|x_t) ~~ q(x_(t-1) | x_t"," x_0) qquad AA t
$$
在训练中，$p_theta(x_(t-1)|x_t)$ 重参数化为
$$
p_theta(x_(t-1)|x_t) = ccN(x_(t-1)"; " mu_theta(x_t,t), sigma_t^2 I)
$$

### 训练

由MLE出发推导出ELBO

![image-20251124204440873](image-20251124204440873.png)

可计算化ELBO

![image-20251120222239108](image-20251120222239108.png)

根据两个高斯分布的KL散度公式，里面只有中间 $mu$ 这一项与 $theta$ 有关（两个高斯分布均值的某种距离），可以简化ELBO里的KL散度为均值之差，
$$
1/(2sigma_t^2)norm(tilde(u)_t (x_t,x_0) - mu_theta(x_t,t))_2^2
$$
根据贝叶斯公式计算**后验分布** $q(x_(t-1) | x_t"," x_0)$ 的均值和方差（证明见作业，配方分析法）有
$$
q(x_(t-1) | x_t"," x_0) = ccN(x_(t-1); tilde(u)_t(x_t,x_0),tilde(beta)_t I)
$$
![image-20251124205524834](image-20251124205524834.png)

在均值公式里代入 $x_t = sqrt(bar(alpha)_t)x_0+sqrt(1-bar(alpha)_t)epsi$ ，后验分布的均值可以简化为
$$
tilde(u)_t(x_t,x_0) = 1/(sqrt(1-beta_t)) (x_t-beta_t/sqrt(1-alpha_t)epsi)
$$
通过参数化 $p_theta$ 为类似的形式，我们可以把均值之差进一步简化为噪声之差。
$$
mu_theta(x_t,t) = 1/(sqrt(1-beta_t)) (x_t-beta_t/sqrt(1-alpha_t)epsi_theta(x_t,t))
$$
从而有

![image-20251124205822488](image-20251124205822488.png)

训练方法称作 Denoising Diffusion Probabilistic Model (DDPM)

## 其他重要公式

高维高斯分布、两个高斯分布的KL散度公式推导（作业）

对数正态分布的PDF
$$
f(x) = 1/(sqrt(2pi) sigma x) exp (-(log x-mu)^2/(2sigma^2))
$$
积分的线性性
$$
int_x p_star(x)log D_(theta')(x)dx + int_x p_theta (x) log(1-D_(theta')(x))
$$
上面的积分区间是固定的，与 $D_(theta')(x)$ 无关，最大化一个积分可以转化为最大化这个积分的被积函数。

高维求导 Jacobi 矩阵 **（注意行、列的方向）**

$$
D_g(z) =  (del (g_1,g_2,...,g_d)) / (del (z_1,...,z_d)) 
= [pp (g_1) (z_1), ..., pp (g_1) (z_d); vdots, ddots, vdots; pp (g_d) (z_1), ..., pp (g_d) (z_d) ]
$$
