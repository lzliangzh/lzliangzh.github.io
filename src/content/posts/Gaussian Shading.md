---
tags:
  - watermark
date: 2025-12-19
title: Distribution-preserving Sampling
category: Computer
---

## Mechanism

![[image.png]]

- Watermark Diffusion
	- $s: l xx c/f_c xx h/f_(hw) xx w/f_(hw) -->^"Diffuse" s^d: lxxcxxhxxw$
	- $(c xx h xx w)$ dimension; each represent $l$ bits of the watermark.
- Watermark randomization
	- $m = s^d o+ K ("e.g. ChaCha20")$
	- $m ~ U({0,1})$

## Distribution-preserving sampling

Facts
- $y = "dec"(m_(ij)) in [0,2^l-1]$
- $m ~ U({0,1}) => y ~ U({0,...,2^l-1}) => p(y) = 1/2^l$

Notations
- Let $f(x)$ denotes the PDF of $ccN(0,I)$
- Let $F(x)$ denotes the CDF of $ccN(0,I)$
- $F^(-1)(x)$ denotes the PPF of $ccN(0,I)$

Suppose $z_T^s = g(y)$

Given
- $y ~ U({0,1,...,2^l-1})$
- $z_T^s ~ N(0,I)$


> [!Note] Review: Change of Variables Formula
> $x = g(z) <=> p_X(x) = p_Z(z)|dz/dx|$

> [!warning] Bottleneck
> - Bridge $y~U$ with $z~ccN$
> - $U$ is a discrete distribution

> [!Note]  Dequantization & Normalizaion
> Let $y_c = (u + y)/2^l, u ~ N(0,1)$

$y_c ~ U(0,2^l/2^l) = U(0,1)$, thus $p_(Y_c)(y_c) = 1$

$p_Z(z) = p_(Y_c)(y_c) (dy_c)/(dz) => y_c = int p_Z(z) dz = F_Z(z)$

Thus.

> [!note] Theorem  (Inverse Transform Sampling)
> $z = F^(-1)_Z(y_c) = F^(-1)((u+y)/(2^l)) quad "i.e. "  z_T^s = "ppf"((u+i)/2^l)$
 
 Extract the watermark from image:

$$
y = floor( 2^l * F(z_T^s) )
$$

> [!note] Equivilent: in a discretization view
> $y = i <=> z_T^s$ falls into the $i$-th interval $(i/2^l, (i+1)/2^l]$ of  $f(x)$


## Reproduction

- Implement `--rotate` distortion method
- Set `--num 500` 
- Set `--rotate 75` (aligned to Tree-Ring)

Discovery: 
- aligned with the results in the paper
- robust to crop
- **sensitive to rotation**

| **Transformation**             | **tpr_detection** | **tpr_traceability** | **mean_acc** | **std_acc** |
| ------------------------------ | ----------------- | -------------------- | ------------ | ----------- |
| **Median Blur**                | 1.0               | 1.0                  | **0.9987**   | 0.0089      |
| **Resize**                     | 1.0               | 1.0                  | **0.9961**   | 0.0162      |
| **JPEG Compression**           | 1.0               | 1.0                  | **0.9884**   | 0.0252      |
| **Gaussian Blur**              | 1.0               | 1.0                  | **0.9841**   | 0.0246      |
| **Random Crop**                | 1.0               | 1.0                  | 0.9742       | 0.0164      |
| **Random Drop**                | 1.0               | 0.99                 | 0.9614       | 0.0373      |
| **Gaussian Std**               | 1.0               | 1.0                  | 0.9558       | 0.0594      |
| **Brightness**                 | 0.98              | 0.95                 | 0.9532       | 0.0939      |
| **S&P Noise**                  | 1.0               | 0.98                 | 0.9364       | 0.0664      |
| **==Rotate (Our Discovery)==** | **0.0**           | **0.0**              | **0.5005**   | 0.0335      |

Environment Configuration:

```
python 3.9
skimage==0.0 => scikit-image
transformers
huggingface_hub
```

## Why Robust to Crop & Scale: Vote

> Similar to voting, if the bit is set to 1 in more than half of the copies, the corresponding watermark bit is set to 1; otherwise, it is set to 0. This process restores the true binary watermark sequence s′.

假设图像被剪掉了 50%，这意味着有一半的水印副本丢失了。但由于 GS 在全图重复嵌入了多份副本，只要剩下的 50% 区域中，识别出的 Bit 1 比例依然占优（>1/2），根据上述的 Voting 规则，最终提取的原始比特流依然能被完美还原。

## Why Sensitive to Rotation

- 旋轉徹底改變了所有塊的**相對朝向和內部索引順序**。解碼器在 $(x, y)$ 坐標系下看到的數據已經被重新洗牌，無法通過簡單的位移搜索來對齊。
- 在 **空域（Spatial Domain）** 操作统计分布。
	- 追求“分布不失真”（Performance-Lossless），这使得它必须深入到每一个具体的噪声采样点。这种对“点”的极致追求，牺牲了对几何变换（如旋转）的宏观鲁棒性。
- 频域的介入可以比较好地处理旋转
