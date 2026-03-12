---
title: 强化学习 | 02 Actor-Critic
date: 2025-12-09
tags:
  - RL
category: Computer
---
## TL; DR

PPO不止在目标函数上优化了策略梯度方法，还引入了新的算法框架 Actor-Critic 架构，用神经网络**建模价值函数**。

## Actor-Critic 架构

Actor-Critic (AC) 架构是一种强化学习 (RL) 算法的通用框架，混合了策略梯度和 Q-learning (价值函数估计) 的思想。Actor-Critic 在原始策略梯度架构的基础上，用神经网络建模价值函数（即「价值网络」）

策略网络（Actor，执行者）负责学习和输出策略 $pi(a|s)$，即在给定状态 $s$ 下采取动作 $a$ 的概率分布。训练时，通过 Actor-Critic 架构中 Critic 提供的优势函数或价值估计来更新策略，目标是提高能获得更高奖励的动作的概率。

$$
L^"CLIP"(theta)  = hat(bbbE)_tau[min(r_t(theta) hat(A)_t, "clip"{r_t(theta)}hat(A)_t)]
$$
	
价值网络（Critic，评论家）负责估计价值函数，目标是准确地预测给定状态的价值 $V(s)$，用于评估当前 Actor 所采取策略的好坏。这通常是一个回归问题。训练时，通过时间差分 (Temporal Difference, TD) 学习方法，最小化其价值估计与实际观察到的回报之间的误差。

$$
L(\phi) = \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - \hat{R}_t)^2 \right]
$$
这里，$\hat{R}_t$ 是用于训练 Critic 的**目标回报 (Target Return)**。这可以是实际的累积奖励、TD 目标、或 广义优势估计 (GAE) 等。

因此，PPO算法中的**总损失函数**即为

$$
\text{Minimize} \left[ \overbrace{-L^{CLIP}(\theta)}^{\text{Policy Term}} + \overbrace{c_1 L^{VF}(\phi)}^{\text{Value Term}} + \overbrace{-c_2 S(\pi_\theta)}^{\text{Entropy Term}} \right]
$$

熵 $L_{entropy}$（对应 $S(\pi_\theta)$）是可选的正则化项，用于增加策略的探索性。它通过在策略损失中添加一个项来最大化策略 $\pi$ 的熵。


> [!NOTE]
> 工作流程简述： 
Rollout Reward Advantage
> 
> 1. Actor 在当前状态 $s$ 下根据策略 $\pi$ 选择一个动作 $a$。
> 2. 执行动作 $a$，环境返回奖励 $r$ 并转换到下一个状态 $s'$。
> 3. Critic 使用观察到的奖励 $r$ 和下一个状态 $s'$ 的估计价值 $V(s')$ 来计算TD 目标和 TD 误差（或优势函数 $A(s, a)$）。
> 4. Critic 使用 TD 误差来更新自己的价值网络参数。
> 5. Actor 使用 Critic 提供的优势函数 $A(s, a)$ 作为其策略梯度更新的方向和大小的指导（替代了纯 PG 中需要等到完整 Episode 结束后才能计算的蒙特卡洛回报）。

