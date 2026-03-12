---
title: 统计检验方法的线性模型本质
date: 2026-01-16
tags:
  - statistics
description: “Common Statistical Tests Are Linear Models”, 线性混合模型大一统
---
## “Common Statistical Tests Are Linear Models”

这种说法打开了看待二者关系的新视角，但与其实质相比，显得有些骇人听闻了。实际上是，对于特定形式的线性回归模型，最小二乘法可以将模型系数构造为各组之间的均值差。而检验系数的过程与普通假设检验的过程仍然是一致的。只是用线性回归模型统一了诸如 t-test，anova 等的假设检验方法于一体。

---

具体而言，对于不同组别之间差异显著性的各种常见统计检验方法，可以看做一个 0-1 变量（哑变量）线性回归模型，及对其回归系数 $hat(beta)$ 的检验。

$$
y = beta_0 + beta_1x_1 + beta_2x_2 + ... + beta_n x_n + epsi = Xbeta+epsi
$$

如果要比较 $k$ 组数据的组间差异，则需要 $(k-1)$ 元及以上哑变量线性回归模型。其中，$x_(1,...,k-1) in {0,1}$ 。

使用哑变量，是为了确保预测值 $hat (y)$ 始终是该组的样本均值。在连续变量回归中，回归线为了照顾所有的点，往往无法穿过每一个点的均值。但在哑变量回归中，回归线虽然数学上画出来是直线，但我们只关心 0 和 1 这两个点，它不需要照顾中间的取值，因此它可以精准地“踩”在每一组的均值中心。

通过使用哑变量，我们让模型在 $(x_1,...,x_(k-1)) = 0$ 时的预测值 $hat(y)$ 代表第 0 组的取值，在 $x_i = 1$ 时的预测值 $hat(y)$ 代表第 $i$ 组的取值。这样，模型的预测值 $hat (y)$ 就永远与该组的样本均值相同了，也就使得模型系数的预测值 $hat(beta)_i$ 能够反映不同组别之间的均值差，从而对于系数预测值的显著性检验即等价于均值差检验，也就是实现了 t 检验方法的效果。

用上面的办法，我们就得以将线性模型及其系数的检验，与均值差检验统一起来了。所以，零假设是 

$$
H_0: mu_0 = ... = mu_(k-1) <=> beta_1 = ... = beta_(k-1) = 0
$$

要检验这个假设，我们关注以下两个检验：

### F 检验（联合显著性检验/模型线性相关性检验）

$$
F = ("SSR"//k)/("SSE"//(n-k-1)) ~ F(k, n-k-1)
$$

这其实检验的是线性相关性是否显著。其中 SSR 是回归平方和，SSE 是残差平方和。这与单因素 ANOVA 的 MS_between / MS_within 在数学上一致。

> [!note] 回顾
> 因变量总偏差平方和
> $$
> l_(y y) & =sum(Y_i-overline(Y))^2 = underbrace(sum(Y_i-hat(Y_i))^2)_("残差平方和" Q = sum hat(epsilon)^2)
> + underbrace(sum(hat(Y_i)-overline(Y))^2)_("回归平方和" U=hat(b)^2l_(x x)) 
> $$

如果 F 检验显著，则说明组间有差异，即这个自变量对因变量有显著影响。但不知道具体是自变量的哪个取值对因变量有显著影响。这需要逐一对回归系数进行 t 检验。

### 回归系数的 t 检验

线性模型回归系数的检验的意义是，系数越显著，模型的解释力越强，预报区间就越窄（预报越精确）。

在哑变量编码下，经过最小二乘法拟合得到的模型回归系数 $hat(beta_i)(i>=1)$ 反映了 $mu_i-mu_0$ 的差值。对系数进行 t 检验，可以判断是哪些自变量的取值会导致因变量显著变化。

$$
t = (hat(beta)_0) / (SE(hat(beta)_0)) ~ T(2n-2)
$$

其中 SE 是标准误 (standard error)，定义为

$$
SE(hat(beta)) = sqrt(Var(hat(beta)))
$$

与标准差的区别是：标准差描写的是原始数据点的离散程度，而标准误描写的是估计值的不确定性。

---

## 线性混合模型

我们已经发现了 t 检验、ANOVA 检验的本质是一个哑变量线性回归模型

$$
y = Xbeta+epsi
$$

其中，我们对残差项 $epsi$ 有严格的假设：

- $bbbE (epsi)=0$
- 独立同分布：$epsi ~ N(0,sigma^2I)$，这意味着方差-协方差矩阵 $V$ 是一个单位矩阵的倍数。其中，对角线相等代表方差齐性（每个观测的变异程度一样），非对角线为 0 代表独立性（观测值之间没有相关性）。

$$
V = Var(\epsilon) = \begin{bmatrix} \sigma^2 & 0 & \cdots \\ 0 & \sigma^2 & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

但在实际研究中，数据往往存在嵌套或重复测量（比如，同一名学生在不同时间的成绩），导致残差之间存在相关性。为了捕获这种相关性，我们引入**随机效应项** $Zb$：

$$
y = Xbeta + Zb + epsi
$$

### 随机截距模型

在随机截距模型中，$Z$ 矩阵也和 $X$ 矩阵一样是一个 One-hot 矩阵，作用是指明每一个观测值属于哪一个随机效应的水平。$b$ 是一个随机截距列向量。

比如说，有一些观测值来自学校 1，有一些观测值来自学校 2。那么，来自 1 地的观测值都会在固定效应后加上一个随机截距 $b_1$，来自 2 地的观测值都会加上一个随机截距 $b_2$。

```R
# y: 成绩, x: 学习时间, School: 学校ID
model <- lmer(y ~ x + (1 | School), data = df)
```

随机截距模型中 $b$ 是怎么算的？

### 随机斜率模型

假设你要研究“学习时间” (`Hours`) 对“考试成绩” (`Score`) 的影响，且你认为不同学校的学生不仅起点不同（截距），进步速度也不同（斜率）。

```R
# (1 + Hours | School) 表示截距和 Hours 的斜率都随 School 随机变化
model <- lmer(Score ~ Hours + (1 + Hours | School), data = df)
# 或者, `1 +` 可以省略
model <- lmer(Score ~ Hours + (Hours | School), data = df)
```

需要注意的是，随机斜率模型对数据量要求较高。如果每个分组里只有一两个观测点，模型往往会因“过度拟合”而无法收敛。

### 斜率-截距无关模型

有时候会看到这种写法： 

```R
lmer(Score ~ Hours + (Hours || School), data = df)
```

它等价于 `(1 | School) + (0 + Hours | School)`。它强制截距和斜率之间的**相关性为 0**。当模型太复杂无法收敛（Singular fit）时，通常会尝试这种写法来简化模型。

## Reference

Common statistical tests are linear models (or: how to teach stats), https://lindeloev.github.io/tests-as-linear/#1_the_simplicity_underlying_common_tests