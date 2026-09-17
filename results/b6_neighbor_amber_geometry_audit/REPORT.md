# B6 neighbor q1–q2 Amber 参数化与 q2 几何修复

Bohrium 任务 20735563（项目 4377225）于 2026-09-14 完成，退出码 0。原任务 5/5 个状态都生成了 `slu.mol2` 和 `slu.frcmod`，净电荷约为 0。然而 Amber 日志对四个 q2 输入均报出 `Large distance for BOND`；来源是 q2 构建脚本拓扑编辑后没有重排三维坐标，参数化的成功退出不代表输入几何可用。

| 状态 | 原新建 C=C 距离 | 修复后 C=C 距离 | 修复后最长成键距离 | 修复后 cis/优化 QC |
| --- | ---: | ---: | ---: | --- |
| q1 V1–V2 | 正常 | 未改动 | — | 原参数化保留 |
| q2 full reuse | 17.65 Å | 1.344 Å | 1.574 Å | 通过 |
| q2 one-arm reuse | 27.26 Å | 1.344 Å | 1.579 Å | 通过 |
| q2 full switch d1 | 23.05 Å | 1.344 Å | 1.576 Å | 通过 |
| q2 full switch d2 | 5.03 Å | 1.343 Å | 1.575 Å | 通过 |

修复采用 ETKDGv3 多构象生成与 MMFF94 优化，每个分支 4 个构象、固定随机种子。四个选定构象的 MMFF 优化均收敛；3D 重新判定两根内烯键均为 Z/cis，且原子顺序、元素和键连接与原始 q2 输入一致。修复结构已经更新到 `gromacs_dcm_server_bundle/structures/`。提交前检查脚本 `analysis/validate_q2_mol2_geometry.py` 可拦截旧输入。

四状态重算任务 Bohrium Bohr ID 20744510（任务组 7422312）于 2026-09-17 02:52:44 完成，退出码 0。下载包与四套参数已在本地核验：每套均有非空的 `slu.mol2`、`slu.frcmod`、`parameterization_metadata.json`、`sqm.out` 与参数化日志；SQM 报告 `Calculation Completed`，没有 `Large distance for BOND`、错误或缺参标记。四个输出均为 432 原子、446 键，原子顺序与键图同修复输入一致，总电荷约为 0，最长成键距离 1.574–1.578 Å；以保留键图的原输入和 Amber 输出坐标复核，两根内烯键均仍为 Z/cis。AM1-BCC 修正前电荷残差绝对值不超过 0.0071 e，均经记录的均匀归一化归零。

| q2 状态 | Amber 输出最长键 / Å | 新旧逐原子电荷 RMS 差 / e | 最大绝对差 / e |
| --- | ---: | ---: | ---: |
| full reuse | 1.574 | 0.0165 | 0.0992 |
| one-arm reuse | 1.578 | 0.0212 | 0.1780 |
| full switch d1 | 1.575 | 0.0152 | 0.0865 |
| full switch d2 | 1.575 | 0.0156 | 0.0880 |

**状态：q1 与四个修复后的 q2 已有可供下一阶段使用的 Amber 参数化文件；显式 DCM 建盒、GROMACS 拓扑检查、密度门槛和 MD 均未完成。** 原任务的四套长键 q2 参数仍不可使用。下载归档保留在工作区 `results/b6_neighbor_q2_repair_20744510/`；可复用参数已暂存于本仓库忽略的 `gromacs_dcm_server_bundle/gromacs_dcm/runs/<state>/setup/solute_parameters/`。这些原始输出不进入 Git。
