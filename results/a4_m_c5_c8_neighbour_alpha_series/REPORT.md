# 【作废：计算口径与原总结不一致】A4-m-C5 至 C8 相邻臂反应选择性计算总总结

> **请勿引用本文件。** 本文件误用了 `--neighbour-arms-only`，没有严格复刻《A4-m-C5与A4-m-C8反应选择性计算总总结.md》的默认 `all_active_vinyl_pairs` 口径。正确的统一复现结果位于 `results/a4_m_c5_c8_mixed_10ns_unified_exact/`。

> 版本：2026-09-10  
> 模型范围：只考虑同臂及 A4 环上相邻两臂的乙烯基候选；对位/非相邻臂候选不进入概率计算；不讨论 R3。

## 摘要

C5、C6、C7、C8 的 mixed-q2 状态现已全部完成相同协议的 3 条独立、每条 10 ns 显式二氯甲烷无偏 MD。每个体系的主要参数定义为 `alpha_NAC = ΔΔG(reuse-switch)/RT`；alpha<0 表示 reuse，alpha>0 表示 switch。

统一相邻臂口径下，C5 明确偏 reuse；C6 明确偏 switch；C8 明确偏 switch。C7 的总体点估计偏 switch，但三个重复的方向不一致，95% replica-bootstrap CI 跨过 0，因此目前只能记为未定。27 套 NAC 距离/角度阈值对四个体系都给出相同的点估计方向，说明 C7 的不确定性来自独立轨迹之间的构象盆地差异，而不是某一个阈值的选择。

## alpha_NAC 主结果

313.15 K 下 RT=0.6223 kcal mol^-1。rho=exp(-alpha) 是单对 reuse/switch NAC 几率比。

| 体系 | ΔΔG reuse-switch | 95% CI | alpha_NAC | alpha 95% CI | rho reuse/switch | 结论 |
|---|---:|---:|---:|---:|---:|---|
| A4-m-C5 | -3.975 | [-10.045, -3.414] | -6.39 | [-16.14, -5.49] | 594.9 | reuse |
| A4-m-C6 | +1.256 | [+1.007, +14.961] | +2.02 | [+1.62, +24.04] | 0.1328 | switch |
| A4-m-C7 | +0.354 | [-1.515, +9.030] | +0.57 | [-2.43, +14.51] | 0.5663 | switch（重复间未定） |
| A4-m-C8 | +2.021 | [+1.086, +6.040] | +3.25 | [+1.74, +9.71] | 0.03889 | switch |

## C6 与 C7 的新增 10 ns 结果

| 体系 | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | reuse/switch |
|---|---:|---:|---:|---:|
| A4-m-C6 | +1.073 | +0.184 | +1.256 | 0.1328 |
| A4-m-C7 | +0.391 | -0.037 | +0.354 | 0.5663 |

C6 的总差值为正且 CI 全部位于 0 以上，单对 switch/reuse 优势约为 7.53 倍；其中主要贡献来自接触可及性。C7 的单对 switch/reuse 点估计约为 1.77 倍，但 CI 跨 0，不能据此断言真实总体偏好。

## 重复间稳定性

| 体系 | replica 1 ΔΔG | replica 2 ΔΔG | replica 3 ΔΔG | 27 阈值方向 |
|---|---:|---:|---:|---|
| A4-m-C5 | -3.414 | -10.045 | -4.961 | 27/27 一致 |
| A4-m-C6 | +14.961 | +1.007 | +13.094 | 27/27 一致 |
| A4-m-C7 | -1.515 | +9.030 | -0.180 | 27/27 一致 |
| A4-m-C8 | +6.040 | +3.628 | +1.086 | 27/27 一致 |

C6 的三个重复均偏 switch，但两个重复中的 reuse 对几乎从未进入接触区，因此 CI 上界很大；这说明方向稳定，但自由能差的绝对大小仍受稀有事件检测限影响。C7 的 replica 1 和 3 偏 reuse，replica 2 强烈偏 switch，说明 10 ns 内仍保留明显构象盆地记忆。

## same/cross 辅助结果

| 体系 | ΔΔG cross-same | 95% CI | cross/same NAC odds | 结论 |
|---|---:|---:|---:|---:|---|
| A4-m-C5 | -1.718 | [-2.122, -1.469] | 15.8 | cross |
| A4-m-C6 | -1.189 | [-2.410, -0.255] | 6.76 | cross |
| A4-m-C7 | -0.965 | [-1.635, +0.760] | 4.72 | cross（重复间未定） |
| A4-m-C8 | -0.828 | [-1.449, -0.151] | 3.78 | cross |

四个体系的 cross-same 点估计都偏 cross，但真正区分 R2/R4 路径的仍是 cross 内部的 reuse/switch，而不是把 same 与 cross 直接对应到产物。

## 与实验和单参数模型的关系

实验上 C5 以 R2 为主，而 C6、C7、C8 以 R4 为主。相邻臂 mixed-q2 结果在定性分类上得到 C5=reuse、C6=switch、C8=switch；C7 的点估计为 switch，但统计上未定。因而当前数据支持从 C5 到较长臂体系发生 reuse-to-switch 转换。

但是，四个 alpha_NAC 点估计并不随 C 单调增加：C6 的 alpha 高于 C7，且 C7 的重复间差异很大。因此目前可以为每个 spacer 报告一个有效 alpha_NAC，却不能把四点当作已经验证的单一平滑 alpha(C) 规律。若最终目标是只保留 alpha 一个参数，最稳妥的做法是把 C7 标记为区间约束，而不是强行给出精确点值。

早期 2 ns q1/opposite-q2 pilot 中，C6 往往偏 reuse、C7 偏 switch；与本次 mixed-q2 的 C6 switch 结果不同。这说明 alpha 的表观值还可能依赖已经形成的两条反应边属于何种历史。mixed-q2 与 C5/C8 的关键验证状态完全同构，因此本报告将其作为主比较，旧 pilot 只作为状态依赖证据。

## 计算边界

- 数值是无催化剂底物的构象预组织/NAC 自由能代理，不是 Ru 催化反应的活化自由能垒。
- 所有对比都是 per-pair；候选数多重性没有混入 alpha。
- 非相邻臂候选已排除，R3 及终止、副反应等不在本模型范围内。
- 6 个新增任务全部使用逐帧盒矢量处理 PBC，保留帧比例均为 100%。

## 数据文件

- `alpha_neighbour_series.csv`：C5-C8 alpha 主表及重复级数值。
- `cross_same_neighbour_series.csv`：same/cross 辅助表。
- `../a4_m_c6_c7_mixed_10ns_v2_analysis/free_energy_contrasts.csv`：C6/C7 27 阈值完整结果。
- `../a4_m_c6_c7_mixed_10ns_v2_analysis/block_stability/`：500 ps 分块稳定性结果。
