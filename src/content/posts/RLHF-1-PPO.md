---
title: 强化学习 | 01 目标函数
date: 2025-12-08
tags:
  - RL
category: Computer
---
## 强化学习中的两个重要函数

动作价值函数 $Q_\pi(s, a)$ 衡量的是在给定策略 $\pi$ 下，智能体从状态 $s$ 开始，并采取特定动作 $a$ 后，预期能获得的累积折扣奖励（Expected Discounted Return）。（如果我在状态 $s$ 选择了动作 $a$，然后从下一步开始严格遵循策略 $\pi$，我预期能获得多少总回报？）

状态价值函数与动作价值函数的关系：

- 状态价值函数 $V_\pi(s)$ 是在状态 $s$ 下，所有可能动作的 $Q$ 值的加权平均，权重是策略 $\pi$ 选择这些动作的概率，是 $Q_\pi(s, a)$ 的期望。即， 一个状态的价值 $V_\pi(s)$ 等于你在这个状态下，根据策略 $\pi$ 选择所有动作的平均价值。

-  $Q_\pi(s, a)$ 可以用 $V_\pi$ 来递归定义，这是 $Q$ 函数的贝尔曼方程（Bellman Equation）的核心：
  $$
  Q_\pi(s, a) = \mathbb{E} [R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t=s, A_t=a]
  $$
  即，在状态 $s$ 采取动作 $a$ 的价值 $Q_\pi(s, a)$，等于即时奖励 $R_{t+1}$ 加上下一个状态 $S_{t+1}$ 的折扣价值 $\gamma V_\pi(S_{t+1})$。


## Vanilla Policy Gradient

策略梯度是强化学习中最基础的一类方法，它直接学习和优化策略 $pi_theta$。	

### 训练

- 在第 $i$ 次训练迭代中，算法使用上一次迭代得到的当前策略网络 $\theta^{i-1}$。这个策略 $\pi_{\theta^{i-1}}$ 被用来与环境进行交互，通常是进行一整条轨迹（episode）的采样，或者采样固定数量的步骤 $N$。
- **每一步的动作选择：** 在环境的每一步中，策略网络 $\pi_{\theta^{i-1}}$ 接收当前的**状态 $s_t$** 作为输入，然后根据其当前的策略 $\pi_{\theta^{i-1}}(a|s_t)$ 来采样或选择[^1]一个作 $a_t$。

[^1]: 对于离散动作空间，策略网络通常输出一个概率分布 $P(a|s_t)$。算法会根据这个概率分布随机采样（抽样）得到实际执行的动作 $a_t$。对于连续动作空间，策略网络通常输出一个均值 $\mu(s_t)$ 和一个方差 $\sigma(s_t)$，构成一个高斯分布，算法会从 $N(\mu(s_t), \sigma(s_t))$ 这个分布中随机采样得到实际执行的动作 $a_t$。

- **执行与记录：** 选定的动作 $a_t$ 被送给环境执行。环境返回一个新的状态 $s_(t+1)$ 和一个奖励 $r_t$。这一步的状态-动作对 $(s_t, a_t)$ 以及后续计算所需的奖励（如，优势函数）被记录下来。最终得到的数据集即为 $\{s_1, a_1\}, \{s_2, a_2\}, ..., \{s_N, a_N\}$。
- 计算目标函数，利用梯度上升策略最大化目标函数 $J(theta) <=>L^"PG"(theta)$，调整策略参数 $\theta$，使得在状态 $s_t$ 下选择更优的行动 $a_t$ 的概率更大，从而提高整体的累积奖励。

$$
\theta \leftarrow \theta + \eta \nabla L^{\text{PG}}
$$

### 策略梯度定理

强化学习的最原始的目标是**希望一个策略平均来看能够带来更大的总回报**，即最大化策略 $pi_theta$ 下的累积奖励的期望值，也等价于起始状态 $s_0$ 在策略 $pi_theta$ 下的状态价值函数 $V_(pi_theta)(s_0)$
$$
J(theta) = V_(pi_theta)(s_0)= bbbE_(pi_theta) [sum_(t=0)^T gamma^t r_t] := bbbE_(tau～P_theta(tau)) [G(tau)]
$$
（上式我们用 $G(tau)$ 表示轨迹 $tau$ 下的累积奖励 $sum_(t=0)^T gamma^t r_t$）

$pi_theta$ 会产生无数个发生概率各不相同的轨迹 $tau$（概率分布记为 $P_(theta)(tau)$），在不同轨迹下的累积奖励 $G(tau)$ 也不同，因此 $J(theta)$ 是一个非常复杂的期望值。虽然难以计算，但我们可以用大数定律（蒙特卡洛方法）无偏地近似这个期望。

---

理论上我们求 $grad J(theta)$ 并应用梯度上升策略就可以实现最大化这个目标函数，
$$
grad J(theta) = grad sum_tau P_theta(tau)G(tau) =  sum_tau grad P_theta(tau)G(tau)
$$
但问题在于，一条完整轨迹 $tau$ 的概率 $P_theta(tau)$ 是每一步**环境转移概率** $P(s_(t+1)|s_t,, a_t)$ 和智能体**动作选择概率** $pi_theta(a_t|s_t)$ 的连乘
$$
P_theta(tau) = P(s_0)prod_(t=0)^T P(s_(t+1)|s_t,, a_t) pi_theta(a_t|s_t)
$$
由于表达式中含有未知的环境转移概率，因此即便我们**解析地**写出 $P_theta(tau)$ 的完整形式，也因为连乘导致导数解析式中含环境转移概率，由于不知道环境转移概率，导致导数值不确定。即， $grad P_theta(tau)$ 是不可计算的。

---

为了巧妙避开环境转移概率及其导数，我们注意到，如果我们把 $P_theta(tau)$ 改写成 $log P_theta(tau)$ 的形式，就可以把连乘变成加法。由于环境转移概率与 $theta$ 无关，这样它们就会从导数表达式中消失。

又注意到 $grad log P = (del log P)/(del theta) = (del logP)/(delP) (delP)/(del theta) = (grad P)/P$，我们得到了
$$
\nabla P(\tau; \theta) = P(\tau; \theta) \frac{\nabla P(\tau; \theta)}{P(\tau; \theta)} = P(\tau; \theta) \nabla \log P(\tau; \theta)
$$
恰巧地把 $grad J(theta)$ 写成了一个期望的形式
$$
grad J(theta) = bbbE_(tau～P_theta(tau))[grad log P_theta(tau)G(tau)]
$$
轨迹概率可以逐步拆开为单步的形式

$$
grad = bbbE_(tau～P_theta(tau))[(sum_(t=0)^T grad log pi_theta(a_t|s_t))G(tau)]
$$
而同时，考虑到因果性（Causality），虽然 $G(tau)$ 是整条轨迹的回报，但我们知道在时间步 $t$ 采取的动作 $a_t$，只能影响其之后的奖励，而不能影响其之前的奖励。因此，对于在时间步 $t$ 发生的事件 $(s_t, a_t)$ 来说，我们只需要考虑从 $t$ 时刻开始的未来回报，即累积奖励 $G_t$ 即可。
$$
grad = bbbE_(tau～P_theta(tau))[sum_(t=0)^T grad log pi_theta(a_t|s_t)G_t]
$$
这样我们巧妙避开了环境转移概率及其导数。我们终于可以用大数定律（蒙特卡洛方法）无偏地近似这个期望，从而求得 $grad J(theta)$ 的估计值了。

---

我们发现这个表达式其实正好是另一个函数的梯度
$$
L^{\text{PG}}(\theta) = \mathbb{E}_\tau [\log \pi_{\theta}(a_t|s_t) G_t]
$$
因此，在实际实现中，我们优化的目标函数就是它了。我们通过最大化这个目标函数来间接地最大化原始的累积奖励期望 $J(theta)$。

上面的推导过程又称为**策略梯度定理（Policy Gradient Theorem）**。

### 优势函数

即使使用 $G_t$，蒙特卡洛估计的方差仍然非常大。这是因为 $G_t$ 随每一次采样的轨迹而剧烈变化。高方差意味着训练不稳定，收敛速度慢。

因为对于任何不依赖于动作 $a_t$ 的函数 $b(s_t)$，以下恒等式成立：

$$
\mathbb{E}_{\pi_\theta} [\nabla \log \pi_\theta(a_t|s_t) b(s_t)] = 0
$$

这表明在梯度中减去 $b(s_t)$ 不会改变梯度的期望（保持无偏性）。

为了降低方差，我们可以在不改变期望梯度 $\nabla J(\theta)$ 的前提下，引入一个基线函数 $b(s_t)$，并将权重 $G_t$ 替换为 $G_t-b(s_t)$，移除 $G_t$ 中与动作选择无关、只与状态本身有关的随机波动。

理论上，能最大限度降低方差的最优基线就是状态价值函数 $V_{\pi_\theta}(s_t)$。【**推导很复杂**】

$V_{\pi_\theta}(s_t)$ 代表在状态 $s_t$ 下，智能体平均能获得的长期回报。所以将 $G_t$ 替换为 $(G_t - V_{\pi_\theta}(s_t))$。

### 总结：目标函数

因此，**目标函数**为
$$
\text{maximize } L^{\text{PG}}(\theta) = \mathbb{E}_{t} [\log \pi_{\theta}(a_t|s_t) \hat{A}_t]
$$
其中
$$
\hat{A}_t = G_t - b
$$
$$
G_t = sum_(k=0)^(T-t) gamma^k r_(t+k)
$$

- $\hat{A}_t$ 是**优势函数**（Advantage function），它告诉我们在状态 $s_t$ 下采取行动 $a_t$ 比平均行动好多少。
- 优势函数中，$G_t$ 是实际观测到的<u>当前时刻直到回合结束</u>的**累积奖励**，由<u>未来</u>每个状态 $s_t$ 下采取行动 $a_t$ 得到的**逐状态奖励** $r_(t)$和一个折扣因子 $gamma < 1$ 组成，评估**未来可能获得的总体回报**。过去的奖励是不可改变的历史，与我们现在决策的价值无关。折扣因子的存在，则确保了即时奖励比遥远的未来奖励价值更大「现在比未来更有价值」。
- $b$ 是一个**基线**（baseline），通常取 $G_t$ 的**平均值**或其他估计值。这样当优势函数为正时，可以认为该行动比平均行动好，最大化目标函数；反之则是最小化目标函数。引入基线的目的是**降低梯度估计的方差**，从而让训练更稳定。

## Importance Sampling

### On/off-policy 与重要性采样

> On-policy 表示学习的智能体与和环境交互的智能体是同一个。
>
> Off-policy 表示学习采取行动的智能体和与环境交互的智能体是不同的。

**On-policy 方法的数据利用效率低**。主要原因是其数据的“新鲜度”要求极高且不可复用。

- 在 On-policy 学习中，用于更新策略 $\pi$ 的数据，必须是由**当前策略** $\pi$ 自身与环境互动所采集的。策略更新后，数据即刻作废（Staleness），每进行一次策略 $\pi$ 的更新，**旧策略 $pi_"old"$ 采集到的数据就不能再用于训练新策略 $pi_"new"$**。因为如果继续使用，就会违背“更新策略 $\pi$ 的数据，必须是由当前策略 $\pi$ 自身采集”的原则，导致训练的目标和实际数据的分布不一致，从而可能引起偏差（Bias）或高方差（High Variance），甚至使训练不稳定。
- 对于策略 $\pi$ 与环境互动所采集的数据，在强化学习中，通常是指**完整的轨迹（Trajectory）**$(s_1, a_1, r_1, s_2, a_2, r_2, ..., s_N, a_N, r_N)$。在基于梯度（如策略梯度）的方法中，这些数据用于计算策略梯度 $\nabla J(\theta)$ 的期望
- 上面的环境交互通常是强化学习中最耗时的部分，每次更新都需要重新进行大量采样，导致总训练时间很长。

为了用旧策略 $\pi_{\theta_{old}}$（行为策略）的数据来计算新策略 $\pi_{\theta}$（目标策略）下的期望，我们引入重要性采样。

**重要性采样**是一种统计学工具，其核心作用是**允许我们使用一个不同的概率分布（Off-policy）来估计目标概率分布（On-policy）下的期望值**。

假设我们有两个概率分布 $p(x)$ 和 $q(x)$，我们想要计算在分布 $p(x)$ 下的某个函数 $f(x)$ 的期望值 $\mathbb{E}_{p}[f(x)]$。**如果直接从 $p(x)$ 采样困难**，我们可以从另一个更容易采样的分布 $q(x)$ 进行采样，并通过调整权重来获得正确的期望值：
$$
\mathbb{E}_{p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{q}[f(x) \frac{p(x)}{q(x)}]
$$
其中 $\rho(x) = \frac{p(x)}{q(x)}$ 被称为**重要性权重（Importance Weight）**。

**Off-policy 应用：** 我们因此可以引入两个分布：

- **目标分布 $p$：** 是当前要优化的策略 $\pi_{\theta}$（Target Policy）。
- **采样分布 $q$：** 是用于收集数据的策略 $b$（Behavior Policy）。

通过重要性采样，我们可以用由**行为策略 $b$ 采集的数据**（Off-policy 数据）来估计**目标策略 $\pi_{\theta}$ 的期望**，从而实现 Off-policy 学习。

### Off-policy 的策略梯度估计

在 off-policy 学习中，我们从与正在优化的策略不同的其他策略中采样轨迹。像近端策略优化算法（PPO）和广义近端策略优化算法（GRPO）等流行的 PG 的 off-policy 变体，会使用来自 $pi_"old"$ 的轨迹来优化当前策略。Off-policy 的策略梯度估计是

$$
\hat{g}_\text{off-policy} = \frac1N \sum_{i=1}^N\sum_{t=0}^T \frac{\pi_\theta(a_t^{(i)}|s_t^{(i)})}{\pi_{\theta_{old}}(a_t^{(i)}|s_t^{(i)})}\nabla_\theta\log\pi_\theta(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)})
$$

这看起来像是 Vanilla PG 的重要性采样版本。

## TRPO, PPO

从 Off-policy 的策略梯度估计出发，我们可以构造新的目标函数，使得其导数即为这个公式。

TRPO 给出了一种构造方式

$$
"maximize"_theta quad hat(bbbE)_tau[(pi_theta(a_t|s_t))/(pi_(theta_"old")(a_t|s_t))hat(A)_t]
$$

subject to $hat(bbbE)_tau["KL"(pi_theta(cdot |s_t) || pi_(theta_"old")(cdot |s_t))] <= delta$.

然而，TRPO 虽然有理论上的单调改进保证，但其带硬性约束的优化问题计算复杂（需要二阶近似、共轭梯度和线性搜索等）。

---

PPO（Proximal Policy Optimization，近端策略优化）旨在保留 TRPO 限制策略更新幅度的优点，同时大大简化优化过程。

PPO 通过修改目标函数，将 TRPO 的硬性 KL 约束替换为一种软性约束 (PPO-KL) 或截断机制 (PPO-Clip, 最常用)，使其可以使用标准的一阶优化方法（如 SGD 或 Adam）进行优化。

- **PPO-Clip**：引入了一个截断函数，将概率比率 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ 限制在一个范围 $[1-\epsilon, 1+\epsilon]$ 内。
	
	$$
	L^"CLIP"(theta)  = hat(bbbE)_tau[min(r_t(theta) hat(A)_t, "clip"{r_t(theta)}hat(A)_t)]
	$$
	
	其中 $r_t(theta) = (pi_theta(a_t|s_t))/(pi_(theta_"old")(a_t|s_t))$.

- **PPO-KL / PPO-adaptive Penalty**：更直接地模仿 TRPO 的 KL 散度约束，但将其作为目标函数中的惩罚项而不是硬性约束。

  $$
  L^{KL}(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{old}}} \left[ r_t(\theta) A_t - \beta D_{KL}(\pi_{\theta_{old}}(\cdot|s) || \pi_{\theta}(\cdot|s)) \right]
  $$
  
  其中$\beta$ 是一个自适应的惩罚系数。如果新旧策略的平均 KL 散度 $\bar{D}_{KL}$ 大于目标 KL 阈值 $d_{target}$，则增大 $\beta$ 以更严格地惩罚策略变化。


## GRPO

GRPO 是一个更通用的策略优化框架，它推广了 TRPO 和 PPO。它允许使用各种不同的距离度量（不限于 KL 散度）来定义新旧策略之间的信任区域，并提供了一种统一的、可扩展的方法来计算其梯度和更新。关于 GRPO 的介绍，可以参见 HW3 的 [[Archived/trival/README|README]].


**Advantage estimation**. The core idea of GRPO is to sample many outputs for each question from the policy $\pi_\theta$ and use them to compute a baseline. This is convenient because we avoid the need to learn a neural value function $V_\phi(s)$, which can be hard to train and is cumbersome from the systems perspective. For a question $q$ and group outputs $\{o^{(i)}\}_{i=1}^G\sim\pi_\theta(\cdot|q)$, let $r^{(i)}=R(q,o^{(i)})$ be the reward for the $i$-th output. DeepSeekMath and DeepSeek R1 compute the group-normalized reward for the $i$-th output as

$$
   A^{(i)} = \frac{r^{(i)}-mean(r^{(1)},r^{(2)},\cdots, r^{(G)})}{std(r^{(1)}, r^{(2)},\cdots, r^{(G)}) + advantage\_eps}\quad (Eq.28)
$$

where $\texttt{advantage\_eps}$ is a small constant to prevent division by zero. Note that this advantage $A^{(i)}$ is the same for each token in the response, i.e., $A_t^{(i)} = A^{(i)}, \forall t\in 1,\cdots, |o^{(i)}|$, so we drop the $t$ subscript in the following.

**GRPO objective**. The GRPO objective combines three ideas:

   1. Off-policy policy gradient;
   2. Computing advantage $A^{(i)}$ with group normalization;
   3. A clipping mechanism, as in PPO.

The purpose of clipping is to maintain stability when taking many gradient steps on a single batch of rollouts. It works by keeping the policy $\pi_\theta$ from straying too far from the old policy.

The GRPO-Clip objective uses a min function to clip the probability ratio, preventing the policy from deviating too far from the old policy during training.

Let us first write out the full GRPO-Clip objective, and then we can build some intuition on what the clipping does (Eq.29):

$$
\begin{align*}
J_{GRPO-Clip}(\theta) &= E_{q\sim \mathcal D, \{o^{(i)}\}_{i=1}^G \sim \pi_\theta(\cdot|q)}\\&[\frac1G\sum_{i=1}^G \frac1{|o^{(i)}|}\sum_{t=1}^{|o^{(i)|}}\min (\frac{\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})}{\pi_{\theta_{old}}(o_t^{(i)} |q,o_{<t}^{(i)})}A^{(i)}, clip(\frac{\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})}{\pi_{\theta_{old}}(o_t^{(i)} |q,o_{<t}^{(i)})},1-\epsilon, 1+\epsilon)A^{(i)})]
\end{align*}
$$

The hyperparameter $\epsilon>0$ controls how much the policy can change. To see this, we can rewrite the per-token objective in a more intuitive way. Define the function

$$
g(\epsilon, A^{(i)}) = \begin{cases}
(1+\epsilon) A^{(i)} \quad \text{if }A^{(i)}\ge 0\\
(1-\epsilon) A^{(i)} \quad \text{if }A^{(i)} <0
\end{cases} 
$$

We can rewrite the per-token objective as

$$
\text{per-token objective} = \min (\frac{\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})}{\pi_{\theta_{old}}(o_t^{(i)} |q,o_{<t}^{(i)})}A^{(i)}, g(\epsilon, A^{(i)}))
$$

We can now reason by cases. When the advantage $A^{(i)}$ is positive, the per-token objective simplifies to

$$
\text{per-token objective} = \min (\frac{\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})}{\pi_{\theta_{old}}(o_t^{(i)} |q,o_{<t}^{(i)})}, 1+\epsilon) A^{(i)}
$$

Since $A^{(i)}>0$, the objective goes up if the action $o_t^{(i)}$ becomes more likely under $\pi_\theta$, i.e., if $\pi_\theta (o_t^{(i)}|q, o_{<t}^{(i)})$ increases. The clipping with min limits how much the objective can increase. So the policy $\pi_\theta$ is not incentivized to go very far from the old policy $\pi_{\theta_{old}}$.

Analogously, when the advantage is negative, the model tries to drive down $\pi_\theta(o_t^{(i)}|q,o_{<t}^{(i)})$, but is not incentivized to decrease it below $(1-\epsilon)\pi_{\theta_{old}}(o_t^{(i)}|q,o_{<t}^{(i)})$.