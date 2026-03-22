# 基于最优传输（OT）的局部相似度训练自由图像聚类项目

## 1. 项目概述

本项目受到 **LaZSL** 的启发，目标是将其原本用于 **image-attribute matching** 的 **optimal transport（OT）匹配思想**，改造为 **image-image similarity**，并进一步将这种图像间相似度用于 **谱聚类（spectral clustering）**。

核心动机不是继续只依赖 CLIP 的整图全局特征，而是希望探索：

- 是否可以把每张图表示成一个**局部 patch 集合**
- 是否可以通过 **OT** 建模两张图局部区域之间的对应关系
- 是否可以利用这种更细粒度的相似度构建更好的 affinity graph
- 是否最终能够提升无监督图像聚类效果

本项目采用 **training-free** 设定：

- 不重新训练 CLIP
- 不额外训练 adapter
- 不做 prompt tuning

当前 backbone 使用：

- **CLIP（open_clip, ViT-B-32）**

后端聚类使用：

- **sklearn SpectralClustering**

评估指标使用：

- **ACC**（Hungarian matching 后）
- **NMI**
- **ARI**

---

## 2. 项目目标

本项目的长期目标不是机械复现 LaZSL，而是围绕以下几个研究问题展开。

### 2.1 局部信息是否有助于聚类

传统做法通常直接使用 CLIP 的全局 embedding 构图聚类。本项目希望验证：

> 相比纯全局特征，图像的局部区域信息是否能够提供额外的判别信号？

### 2.2 OT 是否比 naive local aggregation 更合理

如果每张图被表示为 patch 集合，那么图像间局部相似度可以有很多种聚合方式，例如：

- patch-patch cosine 后直接平均（avg）
- patch-patch cosine 后做 maxavg
- 使用 OT 建模 patch 集合之间的最优匹配

本项目希望判断：

> OT 是否真的比简单局部聚合更有效？

### 2.3 OT 是核心信号还是 refinement

当前一个重要怀疑是：

> OT 的价值，究竟是能够替代 global similarity，还是只能作为 global similarity 上的 refinement？

这也是当前阶段最关键的问题。

---

## 3. LaZSL 的原始思想与本项目的改写

### 3.1 对 LaZSL 的理解

LaZSL 是一个 **training-free** 的方法，其原始任务是零样本学习。大致流程如下：

1. 为每个类别准备属性集合
2. 对输入图像做随机或多尺度局部裁剪，形成局部图像集合
3. 用 CLIP 编码：
   - 整图全局特征
   - 局部 crop / patch 特征
   - 属性文本特征
4. 构造图像局部与属性之间的 cost matrix
5. 用 Sinkhorn 算法求解 optimal transport plan
6. 用 OT plan 对 similarity 做加权，得到类别分数

关键点在于：

> LaZSL 的 OT 不只是“算一个距离”，而是先求一个最优传输计划 `T`，再用 `T` 去加权相似度。

### 3.2 本项目的改写方向

LaZSL 原本是：

- **image ↔ attribute set**

本项目改成：

- **image A ↔ image B**

具体来说：

- 左边是图像 A 的 patch 集合
- 右边是图像 B 的 patch 集合
- 通过 OT 衡量两个 patch 集合之间的匹配关系
- 再把这种匹配结果转换为图像间 affinity
- 最终送入谱聚类

这意味着本项目不是做分类，而是做：

> **基于 OT 局部匹配的 training-free 图像聚类**

---

## 4. 方法定义与符号约定

### 4.1 全局特征 baseline

对每张图像，仅提取一个全局 CLIP 特征：

$$
g_i \in \mathbb{R}^d
$$

两张图的相似度定义为：

$$
\mathrm{Sim}_{ij}^{global} = \cos(g_i, g_j)
$$

这是当前最基础也是最强的 baseline。

### 4.2 局部 patch 表示

每张图像被表示为一组 patch 特征：

$$
P_i = \{p_{i1}, p_{i2}, \dots, p_{iM}\}
$$

其中：

- $N$：图像总数
- $M$：每张图的 patch 数
- $D$：特征维度

代码中约定：

- `patch_features` 的 shape 为 `(N, M, D)`

当前 patch 提取方式采用：

- **规则网格切分**
- patch 单独送入 CLIP 编码

注意，这里尚未复现 LaZSL 中的随机 crop 机制。

### 4.3 local avg

先定义两张图像 patch-patch 的相似度矩阵：

$$
S_{ij}(m,n) = \cos(p_{im}, p_{jn})
$$

最简单的聚合方式是对所有 patch 对直接平均：

$$
\mathrm{Sim}_{ij}^{avg}
=
\frac{1}{M^2} \sum_{m,n} S_{ij}(m,n)
$$

这个方法的直觉是简单、便宜，但缺点也明显：

- 所有 patch 都同等参与
- 无法区分有信息 patch 与噪声 patch
- 容易被背景和无语义局部干扰

### 4.4 local maxavg

另一种简单聚合方式是：

$$
\mathrm{Sim}_{ij}^{maxavg}
=
\frac{1}{M}\sum_m \max_n S_{ij}(m,n)
$$

这个方法相当于：

- 对图像 A 中每个 patch，找图像 B 中最相似的 patch
- 再对这些最大值做平均

相比纯 avg，它更强调“局部最佳对应”，但依然不具备 OT 的全局匹配约束。

### 4.5 local OT

首先定义 patch-patch 的 cost matrix：

$$
C_{ij}(m,n) = 1 - \cos(p_{im}, p_{jn})
$$

两侧 patch 权重初始设为均匀分布：

$$
a = b = [1/M, \dots, 1/M]
$$

然后使用 Sinkhorn 求解最优传输计划：

$$
T_{ij} = \mathrm{Sinkhorn}(a, b, C_{ij})
$$

得到 OT 距离：

$$
D_{ij}^{OT} = \langle T_{ij}, C_{ij} \rangle
$$

最后将距离转换为 affinity：

$$
A_{ij}^{OT} = \exp(-\gamma (D_{ij}^{OT})^2)
$$

这一方案的核心思想是：

> 不再对所有 patch-patch 相似度做盲目平均，而是通过 OT 自动寻找两组局部之间更合理的匹配结构。

### 4.6 hybrid OT

为了引入全局先验，在 local OT 的基础上进一步构造 hybrid cost：

$$
C_{ij}^{hybrid}(m,n)
=
1 - \left(
\theta \cos(p_{im}, p_{jn}) + (1-\theta)\cos(g_i, g_j)
\right)
$$

其中：

- $\theta$ 控制局部项与全局项的权重
- 当 $\theta$ 较大时，更依赖局部 patch 相似度
- 当 $\theta$ 较小时，更依赖 global prior

此外，还加入了简单的 patch selection 机制：

- 对每张图的 patch，与该图全局特征 $g_i$ 做 cosine
- 按分数排序
- 只保留 top-k patch
- 再进行 OT

其直觉是：

> 让更“贴近整图语义”的 patch 保留下来，减少背景 patch 或低质量 patch 的干扰。

### 4.7 谱聚类设定

当前聚类流程采用：

1. 计算图像两两 affinity matrix
2. 对 affinity 保留 top-k 邻居
3. 做对称化
4. 使用 `SpectralClustering(affinity="precomputed")`

这意味着本项目的关键其实集中在：

> 如何构造一个足够好的图像间 affinity graph

---

## 5. 工程结构

当前代码工程组织如下：

```text
project/
  datasets/
    __init__.py
    cifar10_subset.py
    stl10_subset.py

  backbone.py
  patch_extract.py
  local_similarity.py
  ot_utils.py
  ot_similarity.py
  hybrid_ot_similarity.py
  spectral.py
  evaluate.py

  run_global_cifar10.py
  run_local_avg_cifar10.py
  run_local_ot_cifar10.py

  run_global_stl10.py
  run_local_avg_stl10.py
  run_local_ot_stl10.py
  run_hybrid_ot_stl10.py
```

各模块职责大致如下：

### `backbone.py`

负责加载 CLIP 模型，并提取：

- 全局图像特征
- patch 特征

### `patch_extract.py`

负责将图像切成规则网格 patch。

### `local_similarity.py`

负责实现非 OT 的局部聚合方法，例如：

- avg
- maxavg

### `ot_utils.py`

负责 Sinkhorn 等 OT 基础工具函数。

### `ot_similarity.py`

负责 local OT 相似度 / 距离计算。

### `hybrid_ot_similarity.py`

负责引入 global prior 的 hybrid OT 方案。

### `spectral.py`

负责 affinity graph 后处理以及谱聚类。

### `evaluate.py`

负责 ACC / NMI / ARI 的评估。

---

## 6. 已完成实验与结果总结

### 6.1 CIFAR-10 实验

之所以先从 CIFAR-10 开始，是因为它简单、易验证、便于快速跑通完整流程。

#### 6.1.1 global baseline

实验已成功跑通，且表现明显最好。

这说明：

- CLIP 全局特征是有效的
- 谱聚类流程是健康的
- 数据加载、特征提取、构图、评估代码整体没有明显问题

因此 `global baseline` 可以被视为项目的核心参考上限。

#### 6.1.2 local avg

实验结果很差，远弱于 global baseline。

分析原因：

- CIFAR-10 原图仅 $32 \times 32$
- resize 后再规则切 patch，局部区域语义很弱
- patch 中大量是模糊纹理、背景或低信息区域
- 对所有 patch 对直接平均会引入大量噪声

结论：

> 在 CIFAR-10 上，简单 local aggregation 基本不成立。

#### 6.1.3 local OT

相比 local avg，没有本质改善，仍显著弱于 global。

这个结果说明：

- 当 patch 语义本身很差时
- OT 无法从低质量局部中“凭空创造语义”
- OT 的上限受到局部表示质量的强烈限制

结论：

> CIFAR-10 不适合作为 local OT clustering 的主要验证集。

### 6.2 STL-10 实验

由于 CIFAR-10 分辨率太低，于是将主要验证集切换到 STL-10。

切换原因：

- STL-10 分辨率更高（$96 \times 96$）
- 局部 patch 更可能承载可辨别语义
- 依然是相对标准和易于管理的数据集

#### 6.2.1 global baseline

在 STL-10 上，global baseline 依然最强。

说明：

- 全局 CLIP graph 仍然是最稳定、最强的信号
- 在当前 training-free 聚类设定下，global semantic similarity 非常重要

#### 6.2.2 local avg baseline

相比 CIFAR-10 有所改善，但仍明显弱于 global。

这说明：

- 更高分辨率确实让局部表示变得稍微更有意义
- 但 naive local aggregation 依然不够强

#### 6.2.3 local OT baseline

local OT 明显优于 local avg。

这是一个关键结果，因为它表明：

- OT 比 naive local aggregation 更合理
- patch correspondence 的建模确实带来了收益
- 在局部表示质量尚可时，OT 能提供有效帮助

但它仍然没有超过 global baseline。

因此当前证据更支持：

> OT 可以改善局部相似度建模，但尚不足以替代 global similarity。

#### 6.2.4 hybrid OT

在 local OT 基础上引入 global prior 后，性能继续提升。

这说明：

- global prior 的确有帮助
- 在 local OT 上叠加 global 信息是有效的
- patch selection + global prior 的方向是有意义的

但结果仍未超过 pure global baseline。

因此当前结论是：

$$
\text{global} > \text{hybrid OT} > \text{local OT} > \text{local avg}
$$

至少在当前 STL-10 实验中，这一排序成立。

---

## 7. 当前最重要的实验结论

截至目前，实验结果支持以下几点。

### 7.1 global baseline 是最强信号

无论 CIFAR-10 还是 STL-10，纯全局 CLIP 特征构建的 affinity graph 都表现最稳定且最强。

这意味着：

> 全局语义一致性是当前任务中最可靠的主导信号。

### 7.2 local avg 很弱

说明简单地把图像拆成 patch，再对 patch-patch 相似度做均值聚合，并不能自动带来更好的聚类效果。

这类方法的问题在于：

- 对噪声 patch 太敏感
- 缺乏结构化匹配约束
- 局部语义本身也可能不足

### 7.3 local OT 比 local avg 更好

这一点非常关键。

它说明：

> 在局部表示有效的前提下，OT 的确比 naive local aggregation 更合理。

也就是说，OT 至少不是完全无效的，它在“局部对局部”的建模中确实有正面作用。

### 7.4 hybrid OT 比 pure local OT 更好

说明 global prior 能有效帮助 local OT。

这个结果支持一种更合理的项目定位：

> OT 更像是 global similarity 上的局部 refinement，而不是 global similarity 的替代者。

### 7.5 hybrid OT 仍不如 pure global

这是目前最关键、也最需要认真解释的结果。

它表明：

- 当前局部表示质量还不够强
- OT 虽然有帮助，但帮助有限
- 加入 local 结构后，未必就一定超过全局强基线

---

## 8. 当前未解决的核心问题

### 8.1 OT 的真实贡献尚未被严格证明

虽然 `hybrid OT` 比 `local OT` 更好，但还不能直接得出：

> 提升来自 OT 本身

因为也可能是：

- global prior 带来了主要提升
- OT 只是附带存在
- 即使不用 OT，只要把 global 加回来，效果也差不多

因此，目前还不能严谨地主张：

> OT 在 hybrid setting 中具有独立贡献

### 8.2 当前缺少“有 global、无 OT”的关键对照实验

这是现在最重要的实验缺口。

目前已有：

- `global`
- `local_avg`
- `local_OT`
- `hybrid_OT`

但缺少：

- `hybrid_avg`
- `hybrid_maxavg`

即：

- 保留 local + global 的混合思路
- 但不使用 OT
- 看是否已经能接近 `hybrid_OT`

这是判断 OT 是否真正有额外价值的关键。

### 8.3 纯 local OT 替代 global 的路线当前并不成立

现有证据更支持：

- OT 可以改善局部匹配
- OT 可以作为 refinement
- 但 OT 暂时不能直接替代 global similarity

因此项目叙事上不应继续坚持：

> local OT 可以取代 global baseline

更合理的说法应当是：

> local OT provides a structured local refinement on top of global semantic similarity.

### 8.4 当前局部表示仍然比较粗糙

当前 patch 表示方式是：

- 规则网格切 patch
- patch 单独送入 CLIP 编码

这种方式存在明显局限：

- 规则 patch 不一定对应语义部件
- patch 可能包含大量背景
- patch 的语义信息可能不稳定
- patch 单独编码未必优于直接使用 ViT 内部 token

因此，即便 OT 有潜力，也可能被当前粗糙局部表示限制住。

---

## 9. 当前阶段最重要的下一步

### 9.1 第一优先级：做 OT 贡献归因实验

最优先补齐以下 4 个实验对象：

1. `global`
2. `hybrid_avg`
3. `hybrid_maxavg`
4. `hybrid_OT`

核心研究问题是：

> 在已经引入 global prior 的前提下，OT 是否仍然比无 OT 的 hybrid aggregation 更强？

这是当前阶段最重要的问题。

### 9.2 结果解释标准

#### 情况一：`hybrid_OT ≈ hybrid_avg ≈ hybrid_maxavg`

若结果接近，则说明：

- 主要提升来自 global prior
- OT 的边际贡献很小
- hybrid OT 中的“OT”不是决定性因素

此时项目结论应更保守：

> OT 在当前设定下贡献有限，global prior 才是主要改进来源。

#### 情况二：`hybrid_OT > hybrid_avg / hybrid_maxavg`

若 hybrid OT 稳定优于无 OT 的 hybrid 方法，则可以更有力地说明：

- OT 在 global + local 混合设定中有独立价值
- 这种价值不是简单全局混合就能替代的
- OT 的结构化匹配机制确实提供了额外收益

此时可以更合理地主张：

> OT is a meaningful local correspondence mechanism beyond naive hybrid aggregation.

---

## 10. 后续可能的发展方向

如果归因实验表明 OT 确实有一定作用，但还不足以超过 global baseline，那么项目可以进一步朝以下方向推进。

### 10.1 global + OT fusion

一种更直接的思路是，不在 cost 中混 global，而是在 affinity 层面直接融合：

$$
A^{final} = \alpha A^{global} + (1-\alpha) A^{OT}
$$

优点：

- 更直观
- 更容易解释
- 更方便做消融

可以与 current hybrid OT 做对比，分析哪种融合方式更合理。

### 10.2 global-guided candidate pruning

思路：

1. 先用 global similarity 找到每张图的 top-r 候选邻居
2. 只对这些候选邻居计算 OT
3. 用 OT 做局部 refinement

好处：

- 计算更省
- 符合“global 负责粗筛，OT 负责精修”的叙事
- 更符合当前实验趋势

### 10.3 patch weighting / partial OT

当前 patch 权重是均匀分布：

$$
a=b=[1/M,\dots,1/M]
$$

但实际上不同 patch 的重要性并不一样，因此未来可考虑：

- 非均匀 patch weight
- 基于 global relevance 给 patch 赋权
- partial OT
- unbalanced OT

这可能会让 OT 对噪声 patch 更不敏感。

### 10.4 使用 ViT patch tokens 替代 crop patch 编码

当前 patch 表示来自“切图后单独编码”，但未来可考虑：

- 直接提取 ViT 中间层或最后层 patch tokens
- 将 token 作为局部表示
- 避免重复编码 patch 的额外成本
- 可能保留更一致的语义结构

这有望改善局部表示质量。

### 10.5 更换更适合局部结构的数据集

当前 STL-10 已经比 CIFAR-10 更适合，但仍然不是最理想的局部结构验证集。

未来更适合的数据集包括：

- **CUB**
- **Stanford Dogs**
- **Flowers-102**

这些数据集更强调细粒度局部差异，更可能体现局部匹配与 OT 的价值。

---

## 11. 当前阶段的推荐项目定位

结合现有实验结果，一个更稳妥、也更符合证据的项目定位是：

> 本项目研究如何将 LaZSL 中的 OT matching 思想从 image-attribute setting 迁移到 image-image clustering setting，并分析 OT 在局部图像相似度建模中的实际贡献。

进一步地，当前更合理的核心结论倾向于：

> OT 在当前任务中更可能扮演 global semantic similarity 上的 local refinement mechanism，而不是其直接替代者。

这个定位相比“用 OT 取代全局相似度”更加符合现有实验事实，也更容易支撑后续分析与写作。

---

## 12. 当前的最小必要上下文（用于新会话迁移）

下面这段可以在新对话中作为最小必要背景直接贴出：

---

我在做一个基于 LaZSL 思想的图像聚类项目：把原论文中的 image-attribute OT matching 改成 image-image OT similarity，再把图像两两相似度用于 spectral clustering。

当前实现是 training-free，backbone 用 CLIP（open_clip, ViT-B-32），后端用 sklearn SpectralClustering，评估指标是 ACC / NMI / ARI。

已做的方案包括：

1. **global baseline**  
   每张图只提全局 CLIP embedding，用 cosine similarity 构图做谱聚类。这个 baseline 目前最强。

2. **local avg baseline**  
   每张图切成规则网格 patch，提 patch 特征，两张图的相似度用 patch-patch cosine 做 avg 或 maxavg 聚合。结果明显弱于 global。

3. **local OT baseline**  
   定义 patch-patch cost：
   $$
   C_{ij}(m,n)=1-\cos(p_{im},p_{jn})
   $$
   用均匀 patch 权重做 Sinkhorn OT，得到：
   $$
   D_{ij}^{OT}=\langle T_{ij},C_{ij}\rangle
   $$
   再转 affinity：
   $$
   A_{ij}^{OT}=\exp(-\gamma (D_{ij}^{OT})^2)
   $$
   结果比 local avg 好，但仍明显弱于 global。

4. **hybrid OT**  
   在 cost 中加入 global prior：
   $$
   C_{ij}^{hybrid}(m,n)=1-\left(\theta \cos(p_{im},p_{jn})+(1-\theta)\cos(g_i,g_j)\right)
   $$
   并加入简单 patch selection（按 patch 与自身全局特征的相似度保留 top-k patch）。结果比 pure local OT 好，但仍没超过 pure global。

数据集上：

- CIFAR-10 不适合 local OT，因为分辨率太低、patch 语义太差。
- STL-10 上结果更合理：  
  $$
  \text{global} > \text{hybrid OT} > \text{local OT} > \text{local avg}
  $$

当前最关键未解决问题：

- hybrid OT 的提升是否主要来自 global，而不是 OT 本身？

所以下一步最优先要做的是补齐无 OT 的 hybrid 对照：

- `hybrid_avg`
- `hybrid_maxavg`

然后比较：

- `global`
- `hybrid_avg`
- `hybrid_maxavg`
- `hybrid_OT`

核心目的是判断：

> 在已经引入 global prior 后，OT 是否仍然具有独立贡献？

---

## 13. 当前建议的实验优先级

### 第一优先级

- 实现 `hybrid_avg`
- 实现 `hybrid_maxavg`
- 完成 OT 贡献归因实验

### 第二优先级

- 尝试 `global + OT fusion`
- 尝试 global-guided candidate pruning

### 第三优先级

- patch weighting / partial OT
- 使用 ViT patch tokens
- 切换到更细粒度的数据集

---

## 14. 一句话总结

当前项目最核心的研究问题不是“如何让 local OT 替代 global”，而是：

> **在 training-free 图像聚类中，OT 是否能作为 global semantic similarity 之上的有效局部 refinement，并带来超越 naive hybrid aggregation 的真实增益。**
