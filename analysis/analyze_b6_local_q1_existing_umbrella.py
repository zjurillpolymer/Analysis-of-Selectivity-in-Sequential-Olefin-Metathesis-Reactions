#!/usr/bin/env python3
"""Remap the completed B6 q1 umbrella results to the corrected local question."""
from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/b6_neighbor_q1_umbrella_20260918/analysis/pathway_audit.csv"
OUT = ROOT / "results/b6_local_q1_arm1_competition_20260923"

rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig", newline="")))
selected = []
for r in rows:
    if r["orbit"] == "full_reuse":
        new_pair, channel, degeneracy = "V2-V11", "A1_to_A6", 1
    elif r["orbit"] == "one_arm_reuse":
        new_pair, channel, degeneracy = "V2-V3 (represents V2-V4)", "A1_to_A2", 2
    else:
        continue
    nac = float(r["soft_nac_probability"])
    selected.append({
        "bohr_id": r["bohr_id"], "replica": r["replica"], "old_pair": r["pair"],
        "new_pair": new_pair, "channel": channel, "degeneracy": degeneracy,
        "per_microchannel_nac_weight": nac,
        "degeneracy_weighted_nac": degeneracy * nac,
        "nac_free_energy_kcal_mol": r["nac_free_energy_kcal_mol"],
        "pmf_half_rms_kcal_mol": r["pmf_half_rms_kcal_mol"],
    })
with (OUT / "remapped_existing_umbrella.csv").open("w", encoding="utf-8", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(selected[0])); w.writeheader(); w.writerows(selected)

by_rep = {}
for r in selected:
    by_rep.setdefault(r["replica"], {})[r["channel"]] = r
summary = {"source": str(SRC), "mapping": {"V3-V12": "V2-V11", "V3-V10": "V2-V3/V2-V4"}, "replicas": {}}
for rep, vals in by_rep.items():
    a6 = vals["A1_to_A6"]["degeneracy_weighted_nac"]
    a2 = vals["A1_to_A2"]["degeneracy_weighted_nac"]
    summary["replicas"][rep] = {"A1_to_A6_weight": a6, "A1_to_A2_weight_including_degeneracy_2": a2, "A6_to_A2_weight_ratio": a6/a2, "P_A6_if_normalized": a6/(a6+a2), "P_A2_if_normalized": a2/(a6+a2)}
(OUT / "remapped_existing_umbrella.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
(OUT / "REMAP_REPORT.md").write_text("""# B6 q0 后 arm1 局部分流：已有 umbrella 重标\n\n旧 umbrella 的 `V3–V12` 在当前臂定义下对应 `V2–V11`（A1→A6）；旧 `V3–V10` 对应 `V2–V3`（A1→A2），而 `V2–V4` 是化学等价的第二个微观端点，因此 A2 权重乘以2。\n\n已有任务均为25距离窗口、100 ps窗口平衡+500 ps生产，且任务完成。这里没有重新跑 umbrella，只做编号重标和简并加权。\n\n注意：A1→A2 的两个独立 replica 仍有很大的 basin 差异，因此下面的归一化比例只能作为现有数据的诊断，不能当作稳健最终概率。\n\n计算公式：\n\n`W_A6 = NAC(V2-V11)`；`W_A2 = 2 × NAC(V2-V3)`；`P_A6 = W_A6/(W_A6+W_A2)`。\n""", encoding="utf-8")
print(json.dumps(summary, indent=2))
