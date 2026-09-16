# Sequential Olefin Metathesis Selectivity: A4 and B6

本仓库整理近期的 A4/B6 分子内烯烃复分解选择性工作，保留结构、Amber/GROMACS/umbrella 脚本与协议、精简结果和审计记录。旧三参数拟合、参数扫描、原始轨迹、检查点、下载包和已标作废的报告不在仓库内。

## 分子与状态

| 体系 | 已收入的状态 | 当前用途 |
| --- | --- | --- |
| A4-m-C5/C6/C7/C8 | 各自 q0、q1 cis、q2 adjacent/opposite/mixed/reuse；另保存 q1 trans 候选结构 | q1 三分支、q2 匹配与 R2/R4 产率分析 |
| A4-p-C6/C8 | 各自 q0、q1 cis、q2 adjacent/opposite cis | A4-p 选择性与构象分析 |
| B6-p-C6 | q0、q1、q2 邻臂分支及已构建的 q3 | B6 邻臂约化模型 |
| B6-p-C5/C8 | q0 起始结构 | 后续扩展的输入；尚无同级别的 q1/q2 结果 |

A4 的四臂几何不采用 B6 的“间位臂对排除”规则。B6 模型则固定 q0 为跨臂，并在后续反应图中只考虑相邻臂候选；这是一项模型约定，不等同于 MD 中完全不存在其他接触。A4 的目标产率报告只在 R2/R4 间归一化，详见下列报告，不能把表中的条件概率直接当作全部湿实验产物的绝对产率。

B6 neighbor q1–q2 首批 Amber 参数化已完成，但四个 q2 输入有非物理长键；本地已修复 q2 cis 几何，仍待重新参数化，不能使用首批 q2 参数直接跑 MD。参见 [`B6 Amber 几何审计`](results/b6_neighbor_amber_geometry_audit/REPORT.md)。

## 目录

- [`analysis/`](analysis/)：A4/B6 结构构建、Bohrium 准备与提交、显式 DCM、无偏 MD、umbrella 与结果分析脚本。部分早期脚本保留作溯源；解释结果时优先参考下列较新的报告。
- [`gromacs_dcm_server_bundle/`](gromacs_dcm_server_bundle/)：结构、44 行合并状态清单、DCM 模型和 GROMACS `.mdp`。清单合并自原始主清单、A4-p C6/C8 清单、A4-m C5/C8 与 C6/C7 q2 补充清单；B3 行未收入。
- [`umbrella_specs/`](umbrella_specs/)：关键 A4 umbrella 批次的 `pathway.json`、`windows.csv`、`index.ndx` 和 `.mdp` 等文本协议。这里只是协议/路径记录，不含起始 `.gro/.cpt`、拓扑或窗口轨迹，因此不能直接在本仓库内继续运行已完成的窗口。
- [`results/`](results/)：精简报告、CSV、JSON 和小型诊断图；没有原始 XTC/EDR。
- [`b6_workflow/bohrium/amber_input/`](b6_workflow/bohrium/amber_input/)：B6 AmberTools 参数化任务输入及提交元数据。Bohrium AccessKey/SSO 凭据不入库。

优先阅读 [`A4-m-C5–C8 R2/R4 统一反应图报告`](results/a4_m_c5_c8_r2_r4_complete_umbrella_update/REPORT.md)、[`A4-m-C6/C7 双证据报告`](results/a4_m_c6_c7_r2_r4_dual_evidence/REPORT.md)、[`C5/C8 q1 二维 umbrella 重加权与采样审计`](results/a4_m_c5_c8_q1_umbrella_2d_3rep_analysis/REPORT.md)，以及 [`A4-p 审计`](results/a4p_detailed_analysis/DETAILED_REPORT.md)。这些报告分别说明口径、适用范围和采样限制；它们不是一个已对四个 A4 分子完全验证的统一产率预测模型。

## 最小检查与使用

在仓库根目录验证状态清单、结构路径及乙烯基对定义：

```bash
python analysis/validate_selectivity_bundle.py
```

生成 B6-p-C6 邻居 q1 orbit 表：

```bash
python analysis/build_b6_neighbor_orbit_table.py
```

在具备 AmberTools、GROMACS、Packmol、ParmEd 以及相应原始起点的环境中，显式 DCM 建模以 `gromacs_dcm_server_bundle` 为 `--root`，并通过 `--states` 指定所需状态：

```bash
python analysis/setup_explicit_dcm_gromacs.py \
  --root gromacs_dcm_server_bundle \
  --states A4_m_C6_q0 B6_p_C6_q1_V1_V2_cis \
  --dry-run
```

原始 XTC/TRR/EDR/CPT/TPR/GRO、Bohrium 下载包和缓存需要从本地归档或原计算任务获取；仓库中的脚本与协议不表示仅凭本仓库即可重建所有已计算数值。
