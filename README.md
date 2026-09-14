# B6 邻居约化模型：反应路径与显式 DCM 模拟

这个仓库只保存最近完成的 **B6_p_C6 邻居臂模型**工作：q0→q1→q2（并保留已完成的 q3 分支）、结构构建与质量审计、显式二氯甲烷（DCM）Amber/GROMACS 流程，以及 Bohrium 提交模板。

核心约定是：

- q0 固定为跨臂反应；
- 后续只允许相邻臂乙烯基对，间位候选不进入反应图；
- q1 的候选 orbit 保留简并度，q2 的 `full_reuse`、`one_arm_reuse`、`full_switch_d1`、`full_switch_d2` 分开处理；
- 双键结构默认采用 cis 构型；
- 无偏 MD 用于路径/构象检查，umbrella 或其他自由能计算作为路径解释证据；
- 本仓库不包含旧的三参数拟合、参数扫描、历史 A4/C5/C8 分析，也不包含原始轨迹和大体积压缩包。

## 目录

- `analysis/`：B6 邻居 orbit、q1/q2/q3 结构生成、显式 DCM 建模和轨迹分析脚本。
- `gromacs_dcm_server_bundle/`：可复用的 DCM 模型、GROMACS 参数、B6-only `state_manifest.csv` 和提交所需结构。
- `results/b6_neighbor_orbit_table/`：q1 邻居候选表与简并度审计。
- `results/structure_qc/`：q1/q2 结构的原子数、键序和电荷等 QC 记录。
- `results/md_contacts/`：q0 50 ns、q1 50 ns、q2 50 ns 和 q3 20 ns 的精简 reader-facing 结果（CSV、报告和小图；不含 XTC）。
- `b6_workflow/bohrium/amber_input/`：当前 B6 邻居分支 AmberTools 参数化任务的可审计输入和提交元数据。

## 常用命令

从仓库根目录生成邻居 q1 orbit 表：

```bash
python analysis/build_b6_neighbor_orbit_table.py
```

在有 q0 轨迹和 RDKit/MDAnalysis 环境时生成 q1/q2 结构：

```bash
python analysis/build_b6_q1_md_structure.py --root .
python analysis/build_b6_neighbor_q2_representatives.py --root .
```

显式 DCM setup 脚本以 `gromacs_dcm_server_bundle` 作为 `--root`：

```bash
python analysis/setup_explicit_dcm_gromacs.py \
  --root gromacs_dcm_server_bundle \
  --states B6_p_C6_q1_V1_V2_cis \
           B6_p_C6_q2_neighbor_full_reuse_cis \
           B6_p_C6_q2_neighbor_one_arm_reuse_cis \
           B6_p_C6_q2_neighbor_full_switch_d1_cis \
           B6_p_C6_q2_neighbor_full_switch_d2_cis \
  --parameterize-only --jobs 1
```

AmberTools 远程输入位于 `b6_workflow/bohrium/amber_input/`；其中的 `run_amber.sh` 只做 GAFF2/AM1-BCC 参数化，Packmol/tleap/ParmEd 留给后续本地或 GROMACS 阶段。AccessKey 不写入仓库，使用 `bohr auth login` 的本机凭据。

## 数据边界

仓库中的数值结果均为可追溯的汇总表或小型审计文件。原始 XTC/TRR/EDR/CPT/TPI、Bohrium 下载包和缓存文件被 `.gitignore` 排除；需要时可依据提交元数据重新下载。
