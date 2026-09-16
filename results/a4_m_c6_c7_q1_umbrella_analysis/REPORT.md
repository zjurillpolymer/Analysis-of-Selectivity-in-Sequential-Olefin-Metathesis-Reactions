# A4-m-C6 / A4-m-C7 q1 umbrella 分析

> 更新：2026-09-12  
> 数据状态：4/4 条代表性路径成功；100/100 个窗口完整  
> A4 口径：正四面体近似下不排除任何所谓“间位臂对”；无偏 MD 使用全部 15 个有效乙烯基对，umbrella 使用 pilot 预选的代表性 reuse/switch 路径

## 摘要

本轮补齐了 C6、C7 q1 的 reuse/switch 距离 umbrella，并对完整窗口轨迹进行了距离—取向二维重加权。统一定义为

\[
\Delta\Delta G_{\mathrm{reuse-switch}}
=\Delta G_{\mathrm{reuse}}-\Delta G_{\mathrm{switch}},
\]

因此负值表示 reuse 更有利，正值表示 switch 更有利。

最重要的结论是：

1. **C6 q1 仍未解决。** 一维 PMF 为 +0.055 kcal mol⁻¹，95% bootstrap CI 跨零；二维 NAC 为 +0.165 kcal mol⁻¹，而且轨迹前后半程从 −0.518 反转到 +0.655 kcal mol⁻¹。它与无偏 MD 的不确定结果共同说明 C6 q1 没有可稳定标定的 reuse/switch 偏好。
2. **C7 q1 的所选路径稳定偏 reuse。** 一维 PMF 为 −0.899 kcal mol⁻¹，95% CI [−1.578, −0.243]；二维 NAC 为 −1.126 kcal mol⁻¹，reuse/switch 几率比为 6.11，前后半程均保持 reuse。
3. C7 的二维差异主要来自 contact（−1.044 kcal mol⁻¹），取向只提供 −0.082 kcal mol⁻¹ 的小修正。这与原 C5/C8 报告中“距离可及性主导、取向次要”的解释一致。

## 1. 任务与路径

| Bohrium ID | 分子 | 类别 | 乙烯基对 | 来源 q1 replica | 窗口 |
|---:|---|---|---|---:|---:|
| 20726912 | C6 | reuse | V1–V7 | 1 | 25 |
| 20726914 | C6 | switch | V3–V4 | 1 | 25 |
| 20726913 | C7 | reuse | V2–V5 | 2 | 25 |
| 20726915 | C7 | switch | V1–V8 | 2 | 25 |

每条路径覆盖 0.35–2.75 nm，间隔 0.10 nm；每窗口平衡 100 ps、生产 500 ps，力常数 500 kJ mol⁻¹ nm⁻²。四条路径共包含 60 ns 窗口采样，其中用于分析的生产轨迹为 50 ns、80,100 帧。

## 2. 一维距离 PMF

接触区为 0.325–0.500 nm，非接触区延伸至各路径的实际直方图支持上限；主结果要求分箱总计数至少为 5。WHAM 在实际采样范围内重新计算，95% CI 来自每条路径的 200 个 trajectory-bootstrap profiles。

| 状态 | ΔG reuse | ΔG switch | ΔΔG reuse−switch | 95% CI | reuse/switch odds | 前半程 / 后半程 |
|---|---:|---:|---:|---:|---:|---:|
| C6 q1 | 2.839 | 2.784 | +0.055 | [−0.682, +0.795] | 0.915 | −0.483 / +0.738 |
| C7 q1 | 1.932 | 2.832 | **−0.899** | [−1.578, −0.243] | **4.24** | −0.795 / −1.317 |

C6 的点估计接近零，CI 跨零且前后半程换向，不能判为 switch。C7 的区间完全低于零，且前后半程同号。

对直方图最低计数 1/5/20 与接触分界 0.45/0.50/0.55 nm 的 9 组组合：C6 的 ΔΔG 为 −0.054 至 +0.082 kcal mol⁻¹，符号随定义变化；C7 始终为 −0.935 至 −0.886 kcal mol⁻¹。因此只有 C7 的路径方向具有阈值稳健性。

## 3. 距离—取向二维重加权

利用每个窗口的 pull distance 和完整溶质轨迹，以 0.002 nm 距离分箱的 WHAM/MBAR 等价自洽方程恢复无偏帧权重。取向门沿用原报告的 45°/45° 软门控。下表中 contact 与 conditional orientation 严格相加为 total NAC。

| 状态 | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | reuse/switch NAC odds | 前半程 / 后半程 |
|---|---:|---:|---:|---:|---:|
| C6 q1 | +0.237 | −0.072 | **+0.165** | 0.767 | −0.518 / +0.655 |
| C7 q1 | −1.044 | −0.082 | **−1.126** | **6.11** | −0.877 / −1.421 |

C6 的取向项略微有利于 reuse，但不足以抵消其很小的接触差异；更关键的是前后半程反向，说明该幅度没有收敛。C7 的 contact 和 orientation 均偏 reuse，其中 93% 的总自由能差来自 contact。

![C6/C7 q1 距离—取向重加权结果](../a4_m_c6_c7_q1_umbrella_2d_analysis/q1_2d_reweight_summary.png)

## 4. 与无偏 MD 的交叉验证

| 状态 | 10 ns 无偏 NAC（全部 15 对） | 一维 umbrella | 二维 umbrella NAC | 判断 |
|---|---:|---:|---:|---|
| C6 q1 | −0.477 [−0.902, +2.334] | +0.055 [−0.682, +0.795] | +0.165 | **未解决** |
| C7 q1 | −0.917 [−0.976, −0.856] | −0.899 [−1.578, −0.243] | −1.126 | 新批次与所选路径均偏 reuse |

C7 三个估计的符号和幅度接近；相应的无量纲 proxy 为：无偏 NAC α = −1.47、一维 umbrella α = −1.45、二维重加权 α = −1.81。C6 三种估计跨越零点，因此不应给出一个确定 α。

这里仍需保留一个边界：C7 的旧 2 ns pilot 曾偏 switch，而本轮 umbrella 从新 10 ns 批次中预选的一条代表性端点启动。umbrella 解决了所选距离路径的稀有接触采样，却不能自动遍历所有与距离正交的慢构象盆地。因此，它显著加强了“新 q1 系综中 C7 reuse 可及”的证据，但不能把旧、新初始系综差异从物理上抹掉。

## 5. 质量控制

- 四条路径均为 Bohrium `completed`、exit code 0；100/100 个生产窗口完整。
- 相邻窗口最小高斯重叠为 0.251–0.368；主直方图支持区没有大尺度内部断裂，最差只有 2 个连续空 bin（0.016 nm）。
- 二维分析的权重有效样本量为 10,691–17,286；MBAR 残差均低于 1.0×10⁻¹¹。
- 由四个乙烯基原子重算的距离与 GROMACS pull coordinate 最大差异为 0.0123 Å。
- 二维图的轻度平滑只用于显示；表中概率和自由能均来自未平滑权重。

## 6. 对总体机理的含义

q1 与既有 q2 opposite 结果不能混成一个状态无关参数。C7 在本轮 q1 所选路径上偏 reuse，但在 q2 opposite 中强烈偏 switch；这表明路径记忆会随已形成反应边数和具体反应图改变。更准确的模型应允许 α 依赖反应状态或局部图环境，而不是只依赖 spacer 长度。

不过，umbrella 仍是路径级证据：每个分子只选择一条 reuse 和一条 switch 路径，而 A4 的完整类别包含更多对称等价或构象不等价的具体乙烯基对。无偏 MD 的全部 15 对统计用于补足类别覆盖；不能把两条代表 PMF 称为全部 reuse/switch 类别的穷举自由能。

这些结果是无催化剂底物的构象预组织自由能代理，不是 Ru 催化反应活化自由能垒。

## 数据文件

- `pathway_contact_free_energies.csv`
- `selectivity_contrasts.csv`
- `contact_definition_sensitivity.csv`
- `wham_recomputed/`
- `../a4_m_c6_c7_q1_umbrella_2d_analysis/q1_2d_reweighted_contrasts.csv`
- `../a4_m_c6_c7_q1_umbrella_2d_analysis/pathway_reweight_summary.csv`
- `../a4_m_c6_c7_q1_umbrella_2d_analysis/q1_2d_surface_source_data.csv`
