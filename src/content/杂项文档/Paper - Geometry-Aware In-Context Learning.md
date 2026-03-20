# Paper Strategy & Outline: Geometry-Aware In-Context Learning

**核心主旨 (Core Theme):** **Geometry $\to$ Generation**. 将研究从“低端 Prompt 调参 (Trial & Error)”拔高到“基于流形几何原理解释与控制生成质量 (Prediction & Mechanism)”。

## 1. 核心叙事线 (The Narrative Arc)

- **现象 (Phenomenon):** LLMs 虽然强大，但在处理多义词时长文本生成容易发生 **语义漂移 (Semantic Drift)** 和 **幻觉 (Hallucination)**。这不是简单的概率问题，而是 **表征流形 (Representation Manifold) 的结构缺陷**。
- **诊断 (Diagnosis，分析层的核心):** 揭示多义词流形在不同条件下的演变：
  - **Scaling Law:** 上下文变长能够使得模型更好地展开流形（不同义项的 Cluster 分得越开）。
  - **Robustness Paradox:** 探讨噪声抑制（REL vs. IRRE）下流形是否坍缩；特别是 Instruct 模型在面对噪声 (IRRE) 时更易发生流形坍缩，这也解释了为何其易进行无关脑补。
- **预测 (Prediction - 连接层的桥梁):** 上下文学习 (ICL) 的本质是让模型在激活空间中定位“任务向量”。多义词歧义性会导致任务向量的扰动。我们提出 **Semantic Ambiguity Score (SAS)**，量化流形质量并**准确预测 ICL 何时失效**。
- **干预 (Intervention - 方法层的解决方案):** 基于预测发现，我们提出 **Manifold-Guided ICL (MGDS)**，不再盲目“抽盲盒”找 Prompt，而是通过计算全局流形的距离与结构，以过滤多义词带来的干扰噪声，从而在源头上抑制语义漂移。

## 2. 关键方法与实现细节 (Methodology Details)

### 2.1 评价指标：Semantic Ambiguity Score (SAS)
利用几何特性定义一个反映语义歧义程度的打分：
$$ SAS(x) = 1 - \frac{d(x, c_{nearest})}{d(x, c_{second\_nearest})} $$
- **逻辑假设:** query embedding $x$ 如果距离其次近的 cluster center 越近（高 SAS，处于边界模糊地带），该点越容易在 ICL 推理时引发坍缩崩溃。
- **验证设计:** 绘制包含 1000 个样本的散点图 (Scatter Plot)：X 轴为 SAS，Y 轴为生成质量（BLEU/Human Eval），展示它们存在**强相关性 ($r > 0.7$)**。

### 2.2 检索策略：Manifold-Guided Demonstration Selection (MGDS)
摒弃传统的 Baseline (Random 或纯看字面匹配的 BM 25)，也摒弃纯粹追求距离而忽略边界噪音的标准 K-NN。
实现步骤：
1. **全局定位 (Clustering):** 先在全局流形上确定 Query 属于哪个具体的 Semantic Cluster，把握“宏观方向”。
2. **去噪过滤 (Filtering):** 剔除该 Cluster 中的**离群点 (Outliers)**和**决策边界模糊的样本 (Boundary)**。
3. **锚点选择 (Selection):** 只在该 Cluster 的 **核心高密度区域 (Core Region)** 选择最具代表性的 $k$ 个无歧义样本作为 Demonstrations。

### 2.3 可解释性分析 (Interpretability Details)
不仅要得出生成效果提高的结论，还需要分析**“为什么好”**（抛弃难以观测且容易失败的跨层 Attention / t-SNE 连线，改用具有坚实可行性的统计测量）：
1. **表征相似度的层级收敛 (Layer-wise Distance Convergence):**
   - **操作:** 计算 Query Token 在每一层的 Hidden State 与**目标正确 Sense 的 Cluster Center** 的余弦相似度。
   - **可视化:** 绘制一张 2 D 折线图。X 轴为模型的 Layer $1$ 到 $L$，Y 轴为上述的相似度/距离。
   - **预期结论:** Baseline (Random/BM 25) 的相似度在深层时往往遭遇瓶颈或掉头；而 **Ours (MGDS)** 因为注入了极其纯净的锚点，相似度能一路爬升并在深层实现更高的收敛值，直接解释了为何生成不易被带偏。
1. **多义词诱导头 (Induction Heads) 的激活分布 (备选/可选):** 如果有闲余算力，可尝试探查某些特定的 Induction Heads，验证我们的 MGDS 是否能激发出比 Baseline 更加显著的目标语义特征拷贝现象。

## 3. 详细论文章节架构 (Detailed Paper Blueprint)

- **Title (暂定):** *Geometric Mechanics of Polysemy in LLMs: Manifold Analysis and Guided Generation*
- **1. Introduction:**
   - Hook: LLM 长文本极易因多义词发生跑题。
   - Gap: 现有研究只看 WSD 黑盒或瞎调 Prompt，缺乏 “Geometry $\to$ Generation” 的因果机制。
   - Contributions: Scaling Law / Robustness 分析，提出 SAS 指标，及 MGDS 检索引导策略。
- **2. Manifold Analysis (精简且聚焦的分析):**
   - 不再发散介绍不同架构（如 encoder vs. Decoder / Model size 等细节，将其移至附录）。
   - 只专注讲解：上下文长度增长对流形扩张的促进作用 (The Geometry of Context Scaling) 与抗噪能力 (The Robustness of Manifold Structure)。
- **3. From Geometry to Generation (论文核心支点 - The Bridge):**
   - 定义下游任务（Sense Definition Generation / Long-form Story Generation）。
   - 展示 Metric Correlation: 画出最重磅的散点图 (SAS vs. 评分)，证明流形的几何特性决定了生成的上限。
- **4. Manifold-Guided ICL (干预手段):**
   - 介绍 MGDS 的三个步骤（Cluster, Filter, Select Core）。
   - 主实验结果：与 Random/BM 25 对比 (BLEU / ROUGE)。证明长文本生成中 Semantic Drift 大幅减少。
   - Ablation: 选 core 点 vs. 选 boundary 点的生成效果对比。
- **5. Conclusion:**
  - 升华到：表征工程（Representation Engineering）是对 LLM 行为控制的有效降维打击手段。

## 4. 行动与避坑要求 (Execution Guidelines)
- **Do 讲机制:** 任何 Prompt 工程设计，背后的落脚点必须是“深层模型机理”（如 Task Vector Perturbation 假说）。
- **Do 高端可视化:** t-SNE 图上必须体现生成成功点 (Successful Generation) 和失败点 (Failed Generation) 的分布（预计 Failed Points 会密集地贴合在流形边界与噪声区）。
- **Don't 讲过程:** 切忌在正文中罗列繁冗的“试错过程”（如测过哪些模板，哪些特殊参数），重点去包装 MGDS 的检索优越性。