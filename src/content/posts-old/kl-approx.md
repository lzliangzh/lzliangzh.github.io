---
title: Approximate KL Divergence
date: 2026-1-6
tags:
  - tricks
link: https://xiaobo-yang.github.io/zh/posts/kl_grad_zh/
category: Computer
---
$$
KL[q,p]= sum_x q(x) log (p(x))/(q(x)) ​= E_(x∼q)​[log (p(x))/(q(x))]
$$

在大语言模型的强化学习训练中，有 Reward Shaping和KL Loss两种引入KL散度约束的方式。
- 前者为直接对 reward 值加上 KL 估计，作为一个新的reward $r <- r - beta * KL$；
- KL Loss 则将 KL 估计量放到 Loss 中，计算 $grad_theta KL(pi_theta||pi_"ref")$，一起进行反向传播.

[Schulman (2020)](http://joschu.net/blog/kl-approx.html) 给出了三种KL散度的估计方式，其中 $k_3$ loss被认为是兼备unbiased和low variance的好估计量。DeepSeek的GRPO算法 [(Guo et al. 2025)](https://arxiv.org/abs/2501.12948) 就使用了这种方式:

$$
\begin{align*}
J_{GRPO-Clip}(\theta) &= E_{q\sim \mathcal D, \{o^{(i)}\}_{i=1}^G \sim \pi_\theta(\cdot|q)}\\
&\left[\frac1G\sum_{i=1}^G \frac1{|o^{(i)}|}\sum_{t=1}^{|o^{(i)|}}\min \left(\frac{\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})}{\pi_{\theta_{old}}(o_t^{(i)} |q,o_{<t}^{(i)})}A^{(i)}, \text{clip}\left(\frac{\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})}{\pi_{\theta_{old}}(o_t^{(i)} |q,o_{<t}^{(i)})},1-\epsilon, 1+\epsilon\right)A^{(i)}\right)\right]
\end{align*}
$$

其中

$$
D_(KL) (pi_theta || pi_"ref") = (pi_"ref"(o_i|q))/(pi_theta(o_i|q)) - log (pi_"ref"(o_i|q))/(pi_theta(o_i|q)) - 1
$$

不过，如果使用KL Loss，我们实际上是通过抽样来对KL散度的梯度做估计，这和[Schulman (2020)](http://joschu.net/blog/kl-approx.html) 的分析有一些区别。在训练中，我们构建了估计量后，将其直接作为loss进行反向传播求导，期望这仍然是一个很好的逼近：

$$
\begin{aligned} \nabla_\theta \widehat{\text{KL}}(X) &\approx \nabla_\theta \text{KL}(\pi_\theta \| \pi_{\theta_{\text{ref}}}) = \nabla_\theta \int \pi_\theta(x) \cdot \log\left( \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)} \right) dx \\ &= \int \nabla_\theta \pi_\theta(x) \cdot \log\left( \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)} \right) + \pi_\theta(x) \nabla_\theta \log \pi_\theta(x) dx \\ &= \int \pi_\theta(x) \cdot \nabla_\theta \log \pi_\theta(x) \log\left( \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)} \right) + \pi_\theta(x) \cdot \frac{1}{\pi_\theta(x)} \nabla_\theta \pi_\theta(x) dx \\ &= \mathbb{E}_{x \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(X) \cdot \log\left( \frac{\pi_\theta(X)}{\pi_{\text{ref}}(X)} \right) \right]. \end{aligned}
$$

但**KL的无偏估计的梯度，未必是KL的梯度的无偏估计**，即： 

$$
\mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \widehat{\text{KL}}(X) \right] \neq \nabla_\theta \mathbb{E}_{\pi_\theta} \left[ \widehat{\text{KL}}(X) \right] = \nabla_\theta \text{KL}(\pi_\theta \| \pi_{\theta_{\text{ref}}}).
$$

事实上，二者之间相差一项。对一般形式的期望 $\nabla_\theta \mathbb{E}_{\pi_\theta}[f_\theta(X)]$，展开可得： 

$$
\begin{aligned} \nabla_\theta \mathbb{E}_{\pi_\theta}[f_\theta(X)] &= \int \nabla_\theta \pi_\theta(x) \cdot f_\theta(x) + \pi_\theta(x) \cdot \nabla_\theta f_\theta(x) dx \\ &= \int \pi_\theta(x) \cdot \nabla_\theta \log \pi_\theta(x) \cdot f_\theta(x) dx + \int \pi_\theta(x) \cdot \nabla_\theta f_\theta(x) dx \\ &= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(X) \cdot f_\theta(X) \right] + \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta f_\theta(X) \right]. \end{aligned}
$$

对于策略生成的样本 $X \sim \pi_\theta(\cdot)$，若分别取 $\widehat{\text{KL}}$ 为 $k_1, k_2, k_3$ 对应的损失，则需注意上述梯度偏差的影响。


## Reference

Yang, Xiaobo. (Mar 2025). Gradient Estimation of KL Divergence in Large Language Model Reinforcement Learning. Xiabo’s Blog.  
[https://xiaobo-yang.github.io/posts/kl_grad/](https://xiaobo-yang.github.io/posts/kl_grad/).

John Schulman [“Approximating KL Divergence.”](http://joschu.net/blog/kl-approx.html) 2020.

Guo et al. [“DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”](https://arxiv.org/abs/2501.12948) 2025.