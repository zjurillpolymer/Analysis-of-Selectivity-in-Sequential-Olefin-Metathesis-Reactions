# A4-m-C6/C7：q1 三分支与 R2/R4 传播（无偏 MD + umbrella）

> 本报告严格按 `A4-m-C5与A4-m-C8反应选择性计算总总结.md` 的 NAC 与自由能口径，
> 但按当前问题只保留 R2、R4；q0 固定为 cross，R1/R3 不进入目标产物归一化。

## 1. 反应图映射

q0 的第一条边固定为 cross。q1 的下一条 cross 边必须分成三类：

1. **full reuse**：再次使用第一次反应涉及的完整臂对，完成后生成 **2 个 R2**；
2. **partial reuse**：只复用第一次反应中的一条臂，进入 adjacent 状态；只考虑 R2/R4 时归为 **1 个 R4**；
3. **full switch**：使用两条此前都未参与反应的臂，进入 opposite 状态；随后可能得到 **2 个 R2** 或 **1 个 R4**。

候选对数分别为 1、8、4；opposite 状态中 R2×2/reuse 与 R4/switch 候选数分别为 2、4。
因此单路径 umbrella 的 NAC 概率必须先乘候选数，才能形成分支权重。

传播式为

```text
N(R2) = 2 p(q1 full reuse) + 2 p(q1 full switch) p(q2 opposite R2x2)
N(R4) = p(q1 partial reuse) + p(q1 full switch) p(q2 opposite R4)
w(R2) = N(R2) / [N(R2) + 2 N(R4)]
```

最后一式使用 M(R4)/M(R2)=2；这也显式计入一次 full-reuse 完成会同时产生两个 R2。

## 2. q1 三分支概率

| 体系 | 证据 | full reuse | partial reuse | full switch |
|---|---|---:|---:|---:|
| C6 | unbiased | 0.152 | 0.572 | 0.276 |
| C6 | umbrella | 0.068 | 0.514 | 0.417 |
| C7 | unbiased | 0.267 | 0.408 | 0.325 |
| C7 | umbrella | 0.338 | 0.480 | 0.182 |

这里的 umbrella 行来自二维距离—取向重加权后的 soft NAC，且已乘 1:8:4 候选多重性。
三个类别现在都使用两个独立 q1 端点等权平均；同一类别的两个端点保持相同乙烯基对，
因此这一批专门检验初始构象记忆，而没有混入候选双键对身份变化。

umbrella 的双端点类别平均自由能结果为：

| 体系 | 对比（左−右） | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | NAC 几率比（左/右） | 前半/后半 total |
|---|---|---:|---:|---:|---:|---:|
| C6 | full reuse minus partial reuse | +0.040 | -0.079 | -0.039 | 1.07 | -0.194/-0.050 |
| C6 | partial reuse minus full switch | +0.309 | -0.008 | +0.301 | 0.62 | +0.162/+0.423 |
| C6 | full reuse minus full switch | +0.349 | -0.087 | +0.262 | 0.66 | -0.033/+0.373 |
| C7 | full reuse minus partial reuse | -1.002 | -0.074 | -1.076 | 5.64 | -1.442/-0.788 |
| C7 | partial reuse minus full switch | -0.179 | +0.007 | -0.173 | 1.32 | +0.144/-0.449 |
| C7 | full reuse minus full switch | -1.181 | -0.068 | -1.249 | 7.44 | -1.298/-1.236 |

结论：C6 三类代表路径仍然接近。二维总 NAC 中 full switch 相对 partial reuse 仅有 0.301 kcal mol⁻¹
（1.62 倍）、相对 full reuse 为 0.262 kcal mol⁻¹（1.52 倍）；对应一维接触 ΔΔG 的 bootstrap 区间均跨 0，
所以不能把这种小优势视为稳健排序。C7 的 full reuse 相对 partial reuse 为 −1.076 kcal mol⁻¹（5.64 倍），
相对 full switch 为 −1.249 kcal mol⁻¹（7.44 倍），前后半程及一维 bootstrap 均保持 full reuse 优势；
partial reuse 与 full switch 只有 −0.173 kcal mol⁻¹且前后半程换号。

同一无偏 MD 端点内的配对结果进一步区分了可重复方向与端点依赖：

| 体系 | 端点 | 对比（左−右） | ΔΔG total NAC | 左/右 NAC 几率比 |
|---|---:|---|---:|---:|
| C6 | 1 | full reuse minus partial reuse | +0.300 | 0.62 |
| C6 | 1 | partial reuse minus full switch | -0.135 | 1.24 |
| C6 | 1 | full reuse minus full switch | +0.165 | 0.77 |
| C6 | 2 | full reuse minus partial reuse | -0.459 | 2.09 |
| C6 | 2 | partial reuse minus full switch | +0.778 | 0.29 |
| C6 | 2 | full reuse minus full switch | +0.319 | 0.60 |
| C7 | 1 | full reuse minus partial reuse | -0.654 | 2.86 |
| C7 | 1 | partial reuse minus full switch | -0.929 | 4.45 |
| C7 | 1 | full reuse minus full switch | -1.583 | 12.73 |
| C7 | 2 | full reuse minus partial reuse | -1.517 | 11.45 |
| C7 | 2 | partial reuse minus full switch | +0.391 | 0.53 |
| C7 | 2 | full reuse minus full switch | -1.126 | 6.11 |

C6 的 full reuse−full switch 在两个端点均为正（+0.165、+0.319 kcal mol⁻¹），即方向上都略偏 full switch；
C7 在两个端点均为负（−1.583、−1.126 kcal mol⁻¹），即稳健偏 full reuse。
但两体系的 partial reuse 与另外两类之间都存在明显端点依赖，因此只能解释为构象盆地差异，不能单独冻结成一个精确常数。

## 3. q2 opposite：R2×2 与 R4

| 体系 | 证据 | R2×2/reuse | R4/switch |
|---|---|---:|---:|
| C6 | unbiased | 0.863075 | 0.136925 |
| C6 | umbrella | 0.636301 | 0.363699 |
| C7 | unbiased | 0.000000 | 1.000000 |
| C7 | umbrella | 0.000046 | 0.999954 |

umbrella 行同样已乘 2:4 候选多重性。对应的单路径二维 NAC 对比为：

| 体系 | ΔΔG(switch−reuse) contact | orientation | total NAC | switch/reuse 单路径几率比 | 前半/后半 total |
|---|---:|---:|---:|---:|---:|
| C6 | +0.643 | +0.136 | +0.779 | 0.286 | +0.820/+0.670 |
| C7 | -5.324 | -0.457 | -5.781 | 1.08e+04 | -5.252/-6.322 |

C7 的 opposite 状态在无偏与 umbrella 中都几乎完全导向 R4；C6 则存在定量差异：
无偏 MD 给出 R2×2=0.863，代表路径 umbrella 给出 0.636。

## 4. R2/R4 传播与实验

下表给出 q1、q2 两层分别采用无偏或 umbrella 证据的交叉组合。实验值只在 R2+R4 内归一化。

| 体系 | q1 | q2 opposite | 预测 R2 wt | 预测 R4 wt | 实验 R2/R4 wt |
|---|---|---|---:|---:|---:|
| C6 | unbiased | unbiased | 0.390 | 0.610 | 0.415/0.585 |
| C6 | unbiased | umbrella | 0.328 | 0.672 | 0.415/0.585 |
| C6 | umbrella | unbiased | 0.429 | 0.571 | 0.415/0.585 |
| C6 | umbrella | umbrella | 0.334 | 0.666 | 0.415/0.585 |
| C7 | unbiased | unbiased | 0.267 | 0.733 | 0.220/0.780 |
| C7 | unbiased | umbrella | 0.267 | 0.733 | 0.220/0.780 |
| C7 | umbrella | unbiased | 0.338 | 0.662 | 0.220/0.780 |
| C7 | umbrella | umbrella | 0.338 | 0.662 | 0.220/0.780 |

最重要的结果是：**按当前“只保留 R2/R4、partial reuse 必归 R4”的严格映射，C6 的无偏→无偏传播得到 R2/R4=0.390/0.610，接近实验 0.415/0.585。** C7 得到 0.267/0.733，正确给出 R4 主产物，但高估 R2；
实验为 0.220/0.780。

全 umbrella 传播得到 C6=0.334/0.666、C7=0.338/0.662。C6 若用新 q1 umbrella 配合无偏 q2，
则得到 0.429/0.571，与实验 0.415/0.585 很接近；差异主要来自 C6 q2 opposite 的代表路径 umbrella。
C7 仍由 q1 full-reuse 代表对的强优势把 R2 高估。原因不是反应图映射错误，而是即使端点重复已经平衡，
umbrella 仍只覆盖每类一个预选乙烯基对；它不能自动成为含多个候选对类别的完整系综平均。

因此当前最稳妥的证据分工是：**无偏 MD 的全对求和决定分支人口；umbrella/二维重加权解释代表路径的
接触与取向自由能来源。** 两者都保留，但不把单路径 PMF 强行替代全体系分支概率。

## 5. 质量控制与尚缺实验

本轮 4 条新路径及此前 4 条 partial-reuse 路径均有 25 个窗口；所有 q1 路径的最小相邻高斯重叠为 0.185，最大窗口中心偏差为 0.125 nm。
二维几何重构中，8 条新增端点路径的 pull/几何距离 RMS 误差最大为 0.0029 Å，
有效样本数最小为 10716。数据文件完整，WHAM/MBAR 收敛。
C7 q2 opposite-reuse 路径的二维重加权有效样本数只有 463，明显低于其他路径；
因此 0.000046 这一具体分支概率不应作高精度解读。不过其 switch−reuse ΔΔG 在前、后半程均小于 −5 kcal mol⁻¹，
且无偏 MD 独立得到近乎全 R4，所以“C7 opposite 导向 R4”的方向性结论仍然稳健。

当前端点覆盖已经平衡为每类两个。剩余限制转为**候选对覆盖**：full reuse 只有一个候选对，已基本覆盖；
partial reuse 有 8 个、full switch 有 4 个候选对，而 umbrella 各只抽取其中一个固定双键对。
若坚持让 umbrella 独立承担定量产物预测，需要继续扩展候选对；若采用“无偏决定人口、umbrella 解释自由能来源”的分工，
目前数据已经足以支持 C6/C7 的主结论，不必立刻增加大量任务。

## 6. 当前结论

1. 用户提出的三分支拓扑映射可以闭合，并且必须把 full reuse 对应的两个 R2 同时计数。
2. C6 的无偏 q1+q2 传播在不引入额外拟合参数时把实验 R2 质量分数重现到 2.5 个百分点以内。
3. C7 的 R4 主产物得到一致预测，但 R2 仍高约 4.7 个百分点；误差主要来自 q1，而不是 q2 opposite。
4. 平衡端点重复后，乘 1:8:4 重数的 umbrella q1 概率为 C6=0.068/0.514/0.417、C7=0.338/0.480/0.182；partial 类在两者中仍最大。
5. 当前 umbrella 已经能控制端点记忆，但候选对覆盖仍不足以取代无偏全对统计作为最终分支人口。
