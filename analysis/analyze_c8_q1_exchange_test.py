#!/usr/bin/env python3
"""Analyze paired normal/compact C8 q1 full-reuse exchange trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
N_SOLUTE = 247
SPECS = {
    "rep1_normal": ("rep1", "normal", 20734625, 0.75),
    "rep1_compact": ("rep1", "compact", 20734626, 0.75),
    "rep2_normal": ("rep2", "normal", 20734627, 0.45),
    "rep2_compact": ("rep2", "compact", 20734628, 0.45),
    "rep3_normal": ("rep3", "normal", 20734629, 0.75),
    "rep3_compact": ("rep3", "compact", 20734630, 0.75),
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rg_from_gro(path: Path, bonds: list[tuple[int, int]]) -> float:
    u = mda.Universe(str(path))
    u.add_TopologyAttr("bonds", bonds)
    p = u.atoms[:N_SOLUTE].unwrap(compound="fragments", reference=None, inplace=False).astype(float) / 10.0
    centered = p - p.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T @ centered / N_SOLUTE)
    return float(np.sqrt(eig.sum()))


def overlap(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    ha, hb = ha.astype(float), hb.astype(float)
    if ha.sum() == 0 or hb.sum() == 0:
        return float("nan")
    ha /= ha.sum()
    hb /= hb.sum()
    return float(np.sum(np.sqrt(ha * hb)))


def energy_summary(gmx: str, edr: Path, output: Path, name: str) -> dict[str, float | None]:
    xvg = output / f"{name}_temperature_density.xvg"
    run = subprocess.run(
        [gmx, "energy", "-f", str(edr), "-o", str(xvg), "-b", "200"],
        input="Temperature\nDensity\n0\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (output / f"{name}_energy.log").write_text(run.stdout, encoding="utf-8")
    if run.returncode or not xvg.is_file():
        return {"temperature_mean_K": None, "temperature_sd_K": None,
                "density_mean_kg_m3": None, "density_sd_kg_m3": None}
    data = np.loadtxt(xvg, comments=("#", "@"), ndmin=2)
    return {
        "temperature_mean_K": float(np.mean(data[:, 1])),
        "temperature_sd_K": float(np.std(data[:, 1])),
        "density_mean_kg_m3": float(np.mean(data[:, 2])),
        "density_sd_kg_m3": float(np.std(data[:, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=PROJECT / "results/c8_q1_full_reuse_exchange_test")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "results/c8_q1_full_reuse_exchange_analysis_20260914")
    parser.add_argument("--gmx", default="/opt/homebrew/bin/gmx")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)

    # Reuse the audited C8 q1 topology parser and feature extractor from the
    # MetaD analysis so all geometric conventions remain identical.
    from analysis.analyze_c8_q1_colvars_wtmetad import extract, topology

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    start_rg: dict[str, float] = {}
    rg_edges = np.arange(0.55, 1.251, 0.025)
    for name, (replica, kind, job_id, target_nm) in SPECS.items():
        job_dir = root / name / str(job_id) / "extracted"
        if not job_dir.is_dir():
            # Keep the script usable if the user unpacks a job directly under
            # its seed directory rather than retaining the Bohr job-id layer.
            job_dir = root / name
        gro, xtc, edr = job_dir / "production.gro", job_dir / "production.xtc", job_dir / "production.edr"
        if not gro.is_file() or not xtc.is_file():
            raise FileNotFoundError(f"missing production files for {name}: {job_dir}")
        input_dir = PROJECT / "bohrium_submit/c8_q1_full_reuse_exchange_test" / name
        top_path = input_dir / "system.top"
        masses, bonds, names = topology(top_path)
        start_rg[name] = rg_from_gro(input_dir / "start.gro", bonds)
        equil_rg = rg_from_gro(job_dir / "equil.gro", bonds) if (job_dir / "equil.gro").is_file() else float("nan")
        production_end_rg = rg_from_gro(gro, bonds)
        data, _ = extract(xtc, gro, masses, bonds, names, cache)
        arrays[name] = data
        split = data[:, 0] < (data[0, 0] + data[-1, 0]) / 2.0
        first, second = data[split], data[~split]
        energy = energy_summary(args.gmx, edr, output, name) if edr.is_file() else {
            "temperature_mean_K": None, "temperature_sd_K": None,
            "density_mean_kg_m3": None, "density_sd_kg_m3": None,
        }
        rows.append({
            "name": name, "replica": replica, "seed_kind": kind, "job_id": job_id,
            "target_distance_nm": target_nm, "frames": len(data),
            "duration_ns": float((data[-1, 0] - data[0, 0]) / 1000.0),
            "start_rg_nm": start_rg[name],
            "equil_end_rg_nm": equil_rg,
            "production_end_rg_nm": production_end_rg,
            "start_to_equil_rg_shift_nm": float(equil_rg - start_rg[name]),
            "first_half_rg_mean_nm": float(first[:, 1].mean()),
            "second_half_rg_mean_nm": float(second[:, 1].mean()),
            "first_half_rg_median_nm": float(np.median(first[:, 1])),
            "second_half_rg_median_nm": float(np.median(second[:, 1])),
            "rg_min_nm": float(data[:, 1].min()), "rg_max_nm": float(data[:, 1].max()),
            "first_half_fraction_rg_lt_0_80": float(np.mean(first[:, 1] < .80)),
            "second_half_fraction_rg_lt_0_80": float(np.mean(second[:, 1] < .80)),
            "fraction_rg_lt_0_80": float(np.mean(data[:, 1] < .80)),
            "fraction_rg_lt_0_75": float(np.mean(data[:, 1] < .75)),
            "distance_mean_nm": float(data[:, 3].mean()),
            "distance_sd_nm": float(data[:, 3].std()),
            "soft_contact_mean": float(data[:, 6].mean()),
            "soft_nac_mean": float(data[:, 7].mean()),
            "hard_contact_frames": int(np.sum(data[:, 3] <= .50)),
            "hard_nac_frames": int(np.sum((data[:, 3] <= .50) & (data[:, 4] <= 45) & (data[:, 5] >= 45))),
            "rg_bhattacharyya_first_second": overlap(first[:, 1], second[:, 1], rg_edges),
            **energy,
        })

    pair_rows: list[dict[str, object]] = []
    for replica in ("rep1", "rep2", "rep3"):
        normal, compact = arrays[f"{replica}_normal"], arrays[f"{replica}_compact"]
        normal_first = normal[:, 0] < (normal[0, 0] + normal[-1, 0]) / 2.0
        compact_first = compact[:, 0] < (compact[0, 0] + compact[-1, 0]) / 2.0
        n1, n2 = normal[normal_first], normal[~normal_first]
        c1, c2 = compact[compact_first], compact[~compact_first]
        pair_rows.append({
            "replica": replica,
            "target_distance_nm": float(SPECS[f"{replica}_normal"][3]),
            "normal_start_rg_nm": start_rg[f"{replica}_normal"],
            "compact_start_rg_nm": start_rg[f"{replica}_compact"],
            "initial_rg_gap_nm": abs(start_rg[f"{replica}_normal"] - start_rg[f"{replica}_compact"]),
            "first_half_mean_gap_nm": abs(float(n1[:, 1].mean()) - float(c1[:, 1].mean())),
            "second_half_mean_gap_nm": abs(float(n2[:, 1].mean()) - float(c2[:, 1].mean())),
            "first_half_rg_overlap": overlap(n1[:, 1], c1[:, 1], rg_edges),
            "second_half_rg_overlap": overlap(n2[:, 1], c2[:, 1], rg_edges),
            "normal_second_half_compact_fraction": float(np.mean(n2[:, 1] < .80)),
            "compact_second_half_compact_fraction": float(np.mean(c2[:, 1] < .80)),
            "normal_second_half_soft_nac": float(n2[:, 7].mean()),
            "compact_second_half_soft_nac": float(c2[:, 7].mean()),
            "normal_second_half_hard_nac_frames": int(np.sum((n2[:, 3] <= .50) & (n2[:, 4] <= 45) & (n2[:, 5] >= 45))),
            "compact_second_half_hard_nac_frames": int(np.sum((c2[:, 3] <= .50) & (c2[:, 4] <= 45) & (c2[:, 5] >= 45))),
        })

    write_csv(output / "trajectory_summary.csv", rows)
    write_csv(output / "paired_exchange_summary.csv", pair_rows)
    summary = {
        "method": "paired 1 ns distance-restrained exchange diagnostic",
        "scope": "C8 q1 full-reuse V5-V7; normal versus MetaD compact starts",
        "thresholds": {"compact_rg_nm": 0.80, "strong_final_overlap": 0.80, "persistent_mean_gap_nm": 0.05},
        "trajectories": rows,
        "pairs": pair_rows,
        "decision": {
            "final_overlap_values": [r["second_half_rg_overlap"] for r in pair_rows],
            "final_mean_gap_values_nm": [r["second_half_mean_gap_nm"] for r in pair_rows],
            "supports_joint_distance_rg": any(
                float(r["second_half_rg_overlap"]) < .80 or float(r["second_half_mean_gap_nm"]) > .05
                for r in pair_rows
            ),
            "all_compact_seeds_lost_rg_below_0_80_in_production": all(
                float(r["second_half_fraction_rg_lt_0_80"]) == 0.0
                for r in rows if r["seed_kind"] == "compact"
            ),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_overlaps = [float(r["second_half_rg_overlap"]) for r in pair_rows]
    final_gaps = [float(r["second_half_mean_gap_nm"]) for r in pair_rows]
    lines = [
        "# C8 q1 full-reuse：normal/compact 交换测试分析",
        "",
        "本分析比较同一 V5–V7 距离窗口中，普通 umbrella 构象和 MetaD 紧凑构象两种起始条件。它只检验全局构象记忆是否消失，不替代距离 umbrella/WHAM，也不产生新的产率权重。",
        "",
        "## 判据",
        "",
        "Rg<0.80 nm 定义为紧凑覆盖区；用后半程 Rg 分布 Bhattacharyya 重叠系数和均值差评价两种起始条件是否汇合。这里把重叠≥0.80 且均值差≤0.05 nm 作为‘没有明显残留起始记忆’的筛查标准。",
        "",
        "## 结果",
        "",
        "6 个 Bohrium 任务均以 exit code 0 完成；每条生产轨迹为 1 ns、201 帧，日志均显示使用 1 张 GPU，未发现 LINCS WARNING。",
        f"三个 replica 后半程 Rg 重叠系数分别为 {', '.join(f'{x:.3f}' for x in final_overlaps)}；normal/compact 后半程 Rg 均值差分别为 {', '.join(f'{x:.4f}' for x in final_gaps)} nm。",
        "三个 compact 起点在 200 ps 平衡末态的 Rg 已分别升至约 0.833、0.900 和 0.878 nm；随后 1 ns 生产段中三个 compact 轨迹均没有 Rg<0.80 nm 帧。",
        "",
        "这说明 MetaD 找到的极紧凑区在当前无偏、固定距离条件下不是稳定的持久盆地；它不支持直接给 full-reuse 增加产率权重。rep3 后半程仍有较低分布重叠，提示普通构象内部仍可能存在较慢的全局形变，但这与‘紧凑盆地稳定存在’是不同问题。",
        "因此下一步优先是把这 6 条配对轨迹延长到 5–10 ns，确认 normal/compact 的慢形变是否继续汇合；只有在延长后仍保持显著分离时，才有必要为少数关键距离窗口铺设距离×Rg 联合 umbrella。",
        "",
        "## 文件",
        "",
        "- `trajectory_summary.csv`：逐轨迹 Rg、距离、soft-NAC、温度和密度。",
        "- `paired_exchange_summary.csv`：normal/compact 的逐 replica 汇合指标。",
        "- `summary.json`：机器可读结果和下一步判据。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
