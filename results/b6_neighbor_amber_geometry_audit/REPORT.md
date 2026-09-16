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

**状态：只完成本地几何修复，没有重新提交 Amber，也没有启动 q2 MD。** 原任务的四套 q2 电荷来自长键输入，不能直接视为经验证的生产参数。独立四状态重算包在 `b6_workflow/bohrium/amber_repair_input/`；q1 无需重复参数化。下载的原始任务归档与日志保留在工作区 `results/b6_neighbor_amber_20735563/`，不进入 Git。
