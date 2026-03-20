



## 方差齐性

方差齐性是指，在比较的多个总体或组中，它们各自的方差是相等的。意味着每组内数据围绕其均值的波动情况是相似的。**也就是在 t 检验、ANOVA 检验都反复提到的这个假设**：

$$
sigma^2_1 = sigma^2_2 = ... = sigma^2_n
$$

这是统计学中一个重要的概念，尤其在进行许多统计分析时是一个关键的前提假设。常见的统计检验，如 t 检验 (两组均值的比较)、ANOVA (比较多组均值) 等，都是基于总体方差相等这个假设来推导检验统计量和确定其分布的。

如果数据方差不齐，那么这些检验统计量的分布就会偏离理论分布，从而导致 p 值的计算不准确，可能会得出错误的统计推断结果。

比如，方差不等时均值的比较 (Behrens–Fisher 问题)，就和方差齐性下的情况在数学上有很大的差别，是需要校正的。

> [!question] 
>  
> 预设两个正态总体是相互独立的。已知 $sigma_1^2 != sigma_2^2$，判断 $H_0: mu_1=mu_2$ 

构造检验统计量 

$$
T = (overline(X)- overline(Y)) / sqrt(S_1^2/n_1 + S_2^2/n_2) ~~ t(m^star)
$$

其中 

$$
m^star = (S_1^2/n_1 + S_2^2/n_2)^2 / (1/(n_1-1)(S_1^2/n_1)^2 + 1/(n_2-1)(S_2^2/n_2)^2)
$$

---








在单样本 t 检验中，$H_0: mu=mu_0$. 构造模型 $y_i-mu_0 = beta_0 + epsi_i$。使用最小二乘法估计系数得 

$$
hat(beta)_0 = bar(y)-mu_0
$$

在独立两样本 t 检验中，$H_0: mu_1=mu_2$. 构造模型 $y_i = beta_0 + beta_1 x_i + epsi_i$。




最小二乘法本质上是让因变量在每个自变量取值上都最小化残差平方和。在统计学中，能使一项平方误差之和最小的统计量就是**算术平均数**。因此最小二乘法实际上帮我们自动构建了一个均值差。在零假设条件下，$hat(beta)_0=0$. 我们检验这个系数的显著性，就是在检验均值差的显著性。
 
对于 $hat(beta)_0$ 的显著性检验：构造统计量 $t = (hat(beta)_0) / (SE(hat(beta)_0))$ 进行 t 检验[^1]. 

由于不存在自变量项，有 $Var(\hat{\beta}_0) = Var(\bar{y}) = \frac{\sigma^2}{n}$ 即  $SE(hat(beta)_0) = hat(sigma)/sqrt(n)$. 代入 $t$ 中可发现，这个 $t$ 与单样本 t 检验使用的 t 是等价的。




使用最小二乘法估计的系数


$$
{hat(beta)_1, =, sum w_i (y_i-bar(y)) " where " w_i = (x_i-bar(x))/(sum(x_i-bar(x))^2);
hat(beta)_0, =, bar(y) - hat(beta)_1 bar(x) ,:}
$$

因为 $x$ 是一个 0-1 分类变量，所以 $hat(beta)_1 = 0$ 

两个系数的方差：
- $Var (hat (beta)_1) = Var(sum w_i y_i) = sum w_i^2 Var(y_i) = sigma^2 sum w_i^2 = sigma^2 / (sum(x_i-bar(x))^2)$
- $Var(hat(beta)_0) = Var(bar(y)) + bar(x)^2 Var(hat(beta)_1) - 2 bar(x) Cov(bar(y), hat(beta)_1) => sigma^2(1/n + bar(x)^2 / (sum(x_i-bar(x))^2) )$
- 其中 
	- $Var (bar (y)) = 1/n^2 sum Var(y_i) = sigma^2/n$
	- $Cov(bar(y), hat(beta_1)) = 0$

> [!note] 回顾：方差的性质
> - $Var(aY) = a^2 Var(Y)$
> - $Var(\sum Y_i) = \sum Var(Y_i)$

可以计算出两个系数的标准误。

[^1]: SE 是标准误 (standard error)，其定义为 $SE(hat(beta)) = sqrt(Var(hat(beta)))$。其与标准差的区别是：标准差描写的是原始数据点的离散程度，而标准误描写的是估计值的不确定性。