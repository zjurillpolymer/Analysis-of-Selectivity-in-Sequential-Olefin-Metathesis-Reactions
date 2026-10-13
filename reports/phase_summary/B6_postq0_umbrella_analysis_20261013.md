# B6 post-q0 corrected umbrella analysis

分析对象是 6 个完成的 Bohrium 任务：3 个独立起点，分别对 arm6/V2--V11 和 arm2/V2--V3 做距离 umbrella。每条路径 13 个窗口，中心为 0.35--2.75 nm、间隔 0.20 nm；每个窗口 1 ns equilibration 和 3 ns production。以下统计只读取 `production_pullx.xvg`，没有把 equilibration 数据混入。

## 完整性和稳定性

- 6/6 任务 exit code 0，6/6 `run_status.json` 为 `completed`。
- 78/78 窗口存在，生产段每窗 6001 帧，覆盖 0--3000 ps。
- 相邻窗口的经验直方图重叠系数：arm2 的最小/中位数/最大值为 0.145/0.181/0.259；arm6 为 0.148/0.178/0.270。覆盖是连续的，但边缘窗口仍应谨慎解释。
- 各窗生产段前后半段均值差的最大绝对值：arm2 0.0406 nm，arm6 0.0351 nm。没有看到系统性的生产段漂移；个别窗口仍有约 0.03--0.04 nm 的统计波动。

## WHAM PMF（合并同一路径的三个起点）

| pathway | PMF minimum | F(0.35 nm) | F(0.75 nm) | F(1.75 nm) | F(2.75 nm) |
|---|---:|---:|---:|---:|---:|
| arm2 / V2--V3 | 2.44 nm | 20.57 | 10.65 | 3.43 | 2.96 |
| arm6 / V2--V11 | 1.20 nm | 9.38 | 1.05 | 1.95 | 5.60 |

能量单位为 kJ mol^-1，相对于各自 PMF 的最低点。以 0.30--0.60 nm 作为近接触区、1.50--2.80 nm 作为远距离区，对 WHAM PMF 做一维 Boltzmann 积分得到：

- arm2：近接触概率约 0.14%，远距离概率约 85.2%；
- arm6：近接触概率约 7.18%，远距离概率约 30.1%。

因此在相同的单臂几何坐标下，arm6/V2--V11 进入近接触区明显比 arm2/V2--V3 更有利。若把 arm2 的两个化学等价乙烯基简并度乘以 2，arm2 的近接触权重约为 0.28%，仍低于 arm6 的约 7.18%。这个数值是几何近接触权重，不应直接等同于最终反应产率；还需要把 q1 的路径定义、反应窗口和后续产物映射接上。

## 限制

WHAM 输出对少数边缘 histogram bin 给出 poor-sampling warning；0.35 nm 附近的 PMF 以及由它导出的近接触积分误差会较大。当前结果最可靠的结论是两条路径的相对趋势和 PMF 盆地位置，而不是小概率近接触的高精度绝对值。下一步可用 bootstrap WHAM、去掉每个 production 段前 0.5--1 ns 重算，并按三个起点分别报告 PMF，检查 seed dependence。

## 数值稳定性审计

所有 78 个 production、78 个 equilibration 和 78 个 ramp 段都正常结束；production/equilibration/ramp 日志中均没有 `LINCS WARNING`、非有限数值或异常退出。以一个代表性 production energy 文件为例，温度平均 313.21 K、RMSD 3.31 K，压力平均 2.70 bar，数据全为有限值。

三个起点的准备阶段能量最小化出现了少量 LINCS warning（分别为 15、6、6、2、2、12 次），但都随后收敛到 Fmax < 100；其中 seed_12500 arm2 的 EM 日志曾出现一次瞬时 `Epot=inf`，之后仍完成收敛，且后续 NVT/NPT、ramp 和所有 production 段没有再出现非有限能量或 LINCS warning。因此 production 数据在数值上是稳定的，但 seed_12500 arm2 的初始 EM 记录应标记为较低置信度；若要作为最终定量结果，建议单独用更小 EM 步长重做该起点并比较 PMF。
