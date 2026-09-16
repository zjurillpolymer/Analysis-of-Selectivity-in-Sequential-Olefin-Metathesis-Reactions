# A4-m-C5–C8：R2/R4 产率的统一反应图分析

> q0 固定为 cross；只在 R2/R4 内比较。R1、R3 不参与目标产物归一化。
> 本报告取代旧的 q1 二分类拼接方式；q1 统一拆成 full reuse、partial reuse、full switch。

## 1. 唯一采用的产物映射

- q1 full reuse：生成 2 个 R2。
- q1 partial reuse：进入 adjacent；在 R2/R4 目标模型中生成 1 个 R4。
- q1 full switch：进入 opposite；q2 reuse 生成 2 个 R2，q2 switch 生成 1 个 R4。

```text
N(R2) = 2 p(full reuse) + 2 p(full switch) p(opposite -> R2x2)
N(R4) = p(partial reuse) + p(full switch) p(opposite -> R4)
w(R2) = N(R2) / [N(R2) + 2 N(R4)]
```

其中 q1 三类候选数为 1:8:4；q2 opposite 的 R2×2:R4 matching 数为 2:4。
由于 2 个 R2 与 1 个 R4 含有相同的单体总质量，质量分数还可严格化简为：

```text
w(R2) = p(q1 full reuse) + p(q1 full switch) p(q2 opposite -> R2x2)
w(R4) = p(q1 partial reuse) + p(q1 full switch) p(q2 opposite -> R4)
```

## 2. 全候选无偏 MD 的分支概率

| 体系 | q1 full reuse | q1 partial reuse | q1 full switch | q2 opposite→R2×2 | q2 opposite→R4 |
|---|---:|---:|---:|---:|---:|
| C5 | 0.197676 | 0.476469 | 0.325855 | 0.702165 | 0.297835 |
| C6 | 0.152056 | 0.571746 | 0.276198 | 0.863075 | 0.136925 |
| C7 | 0.266754 | 0.407953 | 0.325293 | 0.000000 | 1.000000 |
| C8 | 0.038492 | 0.705576 | 0.255932 | 0.000000 | 1.000000 |

C5–C8 的 q1 现在均来自 3×10 ns。所有概率都先在每条重复内对该类全部候选对求和，
再跨重复平均并归一化，因此已经包含候选多重性，不再额外乘 1:8:4。

## 3. 四个体系按同一公式得到的产率

这里先以全候选无偏 MD 作为分支人口证据；同时列出 q1/q2 umbrella 组合，作为受限自由能敏感性检查。实验值均在 R2+R4 内重新归一化。

| 体系 | q2 证据 | 预测 R2 wt | 预测 R4 wt | 实验 R2 wt | 实验 R4 wt |
|---|---|---:|---:|---:|---:|
| C5 | unbiased_all_pairs | 0.426 | 0.574 | 0.758 | 0.242 |
| C5 | umbrella_representative | 0.448 | 0.552 | 0.758 | 0.242 |
| C6 | unbiased_all_pairs | 0.390 | 0.610 | 0.415 | 0.585 |
| C6 | umbrella_representative | 0.328 | 0.672 | 0.415 | 0.585 |
| C7 | unbiased_all_pairs | 0.267 | 0.733 | 0.220 | 0.780 |
| C7 | umbrella_representative | 0.267 | 0.733 | 0.220 | 0.780 |
| C8 | unbiased_all_pairs | 0.038 | 0.962 | 0.171 | 0.829 |
| C8 | umbrella_representative | 0.039 | 0.961 | 0.171 | 0.829 |

### q1 umbrella 分支作为输入

C5/C8 的 q1 umbrella 结果来自每类 2 条独立路径；先把接触 ΔG 转成接触概率，再乘 q1 候选多重性 1:8:4 并归一化。它与无偏 MD 的 all-pair soft-NAC 分支不是同一个统计量，因此只作为独立的自由能敏感性结果。

| 体系 | q1 full reuse | q1 partial reuse | q1 full switch | q2 证据 | 预测 R2 wt | 预测 R4 wt |
|---|---:|---:|---:|---|---:|---:|
| C5 | 0.468027 | 0.246748 | 0.285226 | unbiased_all_pairs | 0.668 | 0.332 |
| C5 | 0.468027 | 0.246748 | 0.285226 | umbrella_representative | 0.687 | 0.313 |
| C6 | 0.068491 | 0.514291 | 0.417218 | unbiased_all_pairs | 0.429 | 0.571 |
| C6 | 0.068491 | 0.514291 | 0.417218 | umbrella_representative | 0.334 | 0.666 |
| C7 | 0.338161 | 0.480058 | 0.181781 | unbiased_all_pairs | 0.338 | 0.662 |
| C7 | 0.338161 | 0.480058 | 0.181781 | umbrella_representative | 0.338 | 0.662 |
| C8 | 0.026306 | 0.505881 | 0.467813 | unbiased_all_pairs | 0.026 | 0.974 |
| C8 | 0.026306 | 0.505881 | 0.467813 | umbrella_representative | 0.027 | 0.973 |

全候选无偏→无偏的更新系列为 C5 0.426、C6 0.390、C7 0.267、C8 0.038（R2 wt 分数）；实验为 C5 0.758、C6 0.415、C7 0.220、C8 0.171。
四个体系均已按同一反应图与同一 3×10 ns q1 采样长度处理；剩余偏差必须由分支采样、反应坐标或模型边界解释，不能再归因于 C5/C8 只有 2 ns。

无偏→无偏结果可进一步拆成以下四条互斥路径：

| 体系 | R2: q1 full reuse | R2: full switch→q2 R2×2 | R4: q1 partial reuse | R4: full switch→q2 R4 |
|---|---:|---:|---:|---:|
| C5 | 0.198 | 0.229 | 0.476 | 0.097 |
| C6 | 0.152 | 0.238 | 0.572 | 0.038 |
| C7 | 0.267 | 0.000 | 0.408 | 0.325 |
| C8 | 0.038 | 0.000 | 0.706 | 0.256 |

C5 的 R2 同时来自 direct full reuse 和 opposite 状态的 reuse matching；C6 也有两条来源。
C7/C8 的 opposite 状态几乎只去 R4，所以它们的 R2 产率近似等于 q1 full-reuse 分支人口。

按独立重复重采样得到的无偏→无偏 R2 wt 95% 区间为：C5 0.250–0.663、C6 0.054–0.663、C7 0.248–0.286、C8 0.000–0.068。区间宽度直接反映三条独立重复之间的盆地差异。

## 4. C5/C8 10 ns 补充实验的含义

C5 的 q1 full-reuse 更新人口为 0.198；C8 为 0.038。C8 因而不再是短轨迹给出的近零值，说明延长采样确实找到了此前缺失的 reuse 构象，但该人口仍不足以单独解释实验 R2。

C5 的三个重复仍给出显著不同的 full-reuse 权重，因此其点估计不能视为严格收敛；C8 三个重复均采到非负但较小的 full-reuse 权重，方向比旧结果稳定。由于 R3 已按模型定义排除，这里的差异不能用改变目标产物归一化来修补。

## 5. C5/C8 q1 umbrella 的结果

C5/C8 的 12 条 q1 umbrella 路径已经全部完成并下载：每个体系的 full reuse、partial reuse、full switch 各 2 条独立路径；每条路径 25 个距离窗口（0.35–2.75 nm，0.10 nm 间隔），每窗平衡 100 ps、生产 500 ps。WHAM 在生产段跳过前 100 ps，并用 200 条 trajectory-bootstrap 曲线给出不确定度。

接触 ΔG（接触区 <0.50 nm 相对于支持的非接触区）如下；数值越低表示该类别进入近接触构象越有利：

| 体系 | full reuse ΔG | partial reuse ΔG | full switch ΔG |
|---|---:|---:|---:|
| C5 | 2.064 | 3.777 | 3.253 |
| C8 | 3.626 | 3.078 | 2.692 |

C5 的 PMF 顺序是 full reuse < full switch < partial reuse，因而自由能模型给出 q1 分支约 0.468/0.247/0.285；C8 的顺序是 full switch < partial reuse < full reuse，给出约 0.026/0.506/0.468。这里的分支已经乘入 1:8:4 候选多重性。

按同一 R2/R4 反应图传播，C5 的 q1-umbrella→q2-umbrella 预测为 R2/R4 = 0.687/0.313，C8 为 0.027/0.973；q2 换用无偏 all-pair 时分别为 C5 0.668/0.332、C8 0.026/0.974。
这些结果说明：C5 的 q1 自由能证据比 3×10 ns 无偏人口更支持 R2，但 C8 仍强烈支持 R4；umbrella 能解释路径接近接触的热力学倾向，却不能替代无偏轨迹对真实动力学分支人口的估计。

因此，C5/C8 的 q1 umbrella 补充实验已经闭环；后续比较应同时保留 `unbiased_all_pairs` 与 `umbrella_representative` 两条证据链，不把二者静默合并。

## 6. 文件

- `branch_probabilities.csv`：四体系、两层反应图的统一分支概率。
- `source_combination_predictions.csv`：所有可用无偏/umbrella 组合的产率传播。
- `yield_pathway_decomposition.csv`：R2、R4 的直接与 q2 间接路径贡献。
- `unbiased_bootstrap.csv`：按独立重复重采样的无偏→无偏 R2 wt 区间。
- `../a4_m_c5_c8_q1_three_class_umbrella_analysis/`：12 条 q1 umbrella 的 WHAM、路径/类别接触自由能和三类对比。
