#!/usr/bin/env python3
"""Add replica/block consensus and contact-persistence statistics for B6 q2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATE = "B6_p_C6_q2_V3_V6_V11_V12_cis"
BLOCK_PS = 10_000.0
CONTACT_A = 5.0
SOFT_CENTER_A = 5.0
SOFT_WIDTH_A = 0.5


def soft_contact(distance_A: np.ndarray) -> np.ndarray:
    z = np.clip((distance_A - SOFT_CENTER_A) / SOFT_WIDTH_A, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return stops - starts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_summary = pd.read_csv(args.audit / "pair_state_summary.csv")
    state_summary = state_summary[state_summary["state"] == STATE].copy()
    pair_columns = state_summary["pair"].tolist()
    run_root = args.workspace / "gromacs_dcm/runs" / STATE

    block_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    replica_scores: dict[int, dict[str, float]] = {}
    replica_top5: dict[int, set[str]] = {}
    replica_top10: dict[int, set[str]] = {}

    for replica in range(1, args.replicas + 1):
        path = run_root / f"replica_{replica:02d}" / "vinyl_distances.csv"
        data = pd.read_csv(path)
        time_ps = data["time_ps"].to_numpy(float)
        dt_ps = float(np.median(np.diff(time_ps)))
        scores = {
            pair: float(np.mean(soft_contact(data[pair].to_numpy(float))))
            for pair in pair_columns
        }
        replica_scores[replica] = scores
        ranked = sorted(scores, key=scores.get, reverse=True)
        replica_top5[replica] = set(ranked[:5])
        replica_top10[replica] = set(ranked[:10])

        for block in range(5):
            lower = block * BLOCK_PS
            upper = (block + 1) * BLOCK_PS
            selected = (time_ps >= lower) & (
                (time_ps < upper) if block < 4 else (time_ps <= upper)
            )
            values = {
                pair: float(np.mean(soft_contact(data.loc[selected, pair].to_numpy(float))))
                for pair in pair_columns
            }
            ranking = sorted(values, key=values.get, reverse=True)
            rank_map = {pair: rank for rank, pair in enumerate(ranking, start=1)}
            for pair in pair_columns:
                block_rows.append(
                    {
                        "replica": replica,
                        "block": block + 1,
                        "start_ps": lower,
                        "end_ps": upper,
                        "pair": pair,
                        "soft_q": values[pair],
                        "rank": rank_map[pair],
                        "top5": rank_map[pair] <= 5,
                        "top10": rank_map[pair] <= 10,
                    }
                )

        for pair in pair_columns:
            distance = data[pair].to_numpy(float)
            lengths = run_lengths(distance < CONTACT_A)
            durations = lengths.astype(float) * dt_ps
            event_rows.append(
                {
                    "replica": replica,
                    "pair": pair,
                    "dt_ps": dt_ps,
                    "contact_frames": int(np.count_nonzero(distance < CONTACT_A)),
                    "contact_fraction": float(np.mean(distance < CONTACT_A)),
                    "events": int(len(durations)),
                    "mean_event_ps": float(np.mean(durations)) if len(durations) else 0.0,
                    "median_event_ps": float(np.median(durations)) if len(durations) else 0.0,
                    "q95_event_ps": float(np.quantile(durations, 0.95)) if len(durations) else 0.0,
                    "max_event_ps": float(np.max(durations)) if len(durations) else 0.0,
                    "events_ge_2ps": int(np.count_nonzero(durations >= 2.0)),
                    "events_ge_5ps": int(np.count_nonzero(durations >= 5.0)),
                    "events_ge_10ps": int(np.count_nonzero(durations >= 10.0)),
                }
            )

    blocks = pd.DataFrame(block_rows)
    events = pd.DataFrame(event_rows)
    blocks.to_csv(args.output / "q2_block_pair_scores.csv", index=False)
    events.to_csv(args.output / "q2_contact_events_by_replica.csv", index=False)

    event_summary = (
        events.groupby("pair", as_index=False)
        .agg(
            contact_fraction_mean=("contact_fraction", "mean"),
            events_total=("events", "sum"),
            mean_event_ps=("mean_event_ps", "mean"),
            median_event_ps=("median_event_ps", "median"),
            q95_event_ps_max=("q95_event_ps", "max"),
            max_event_ps=("max_event_ps", "max"),
            events_ge_2ps_total=("events_ge_2ps", "sum"),
            events_ge_5ps_total=("events_ge_5ps", "sum"),
            events_ge_10ps_total=("events_ge_10ps", "sum"),
        )
    )
    event_summary.to_csv(args.output / "q2_contact_event_summary.csv", index=False)

    consensus_rows: list[dict[str, object]] = []
    for _, row in state_summary.iterrows():
        pair = str(row["pair"])
        values = np.array([replica_scores[r][pair] for r in range(1, args.replicas + 1)])
        pair_blocks = blocks[blocks["pair"] == pair]
        event = event_summary[event_summary["pair"] == pair].iloc[0]
        consensus_rows.append(
            {
                "pair": pair,
                "left": int(row["left"]),
                "right": int(row["right"]),
                "topology": row["topology"],
                "soft_q_mean": float(row["soft_q_mean"]),
                "soft_q_replica_median": float(np.median(values)),
                "soft_q_replica_min": float(np.min(values)),
                "soft_q_replica_max": float(np.max(values)),
                "replica_top5_support": sum(pair in replica_top5[r] for r in replica_top5),
                "replica_top10_support": sum(pair in replica_top10[r] for r in replica_top10),
                "block_top5_count_of_15": int(pair_blocks["top5"].sum()),
                "block_top10_count_of_15": int(pair_blocks["top10"].sum()),
                "block_median_rank": float(pair_blocks["rank"].median()),
                "hard_5A_fraction": float(row["hard_5A_fraction"]),
                "events_total": int(event["events_total"]),
                "max_event_ps": float(event["max_event_ps"]),
                "events_ge_5ps_total": int(event["events_ge_5ps_total"]),
                "first_half_soft_q": float(row["first_half_soft_q"]),
                "second_half_soft_q": float(row["second_half_soft_q"]),
                "split_rhat": float(row["split_rhat"]),
                "combined_effective_sample_size": float(row["combined_effective_sample_size"]),
            }
        )
    consensus = pd.DataFrame(consensus_rows).sort_values("soft_q_mean", ascending=False)
    consensus.insert(0, "mean_rank", np.arange(1, len(consensus) + 1))
    consensus.to_csv(args.output / "q2_pair_consensus.csv", index=False)

    first = state_summary.set_index("pair")["first_half_soft_q"]
    second = state_summary.set_index("pair")["second_half_soft_q"]
    first_top5 = set(first.nlargest(5).index)
    second_top5 = set(second.nlargest(5).index)
    summary = {
        "state": STATE,
        "replicas": args.replicas,
        "blocks": 15,
        "block_duration_ps": BLOCK_PS,
        "contact_threshold_A": CONTACT_A,
        "first_second_half_spearman": float(first.corr(second, method="spearman")),
        "first_second_top5_jaccard": len(first_top5 & second_top5) / len(first_top5 | second_top5),
        "first_half_top5": sorted(first_top5),
        "second_half_top5": sorted(second_top5),
        "top_mean_pair": str(consensus.iloc[0]["pair"]),
        "top_replica_consensus_pairs": consensus[
            consensus["replica_top5_support"] == args.replicas
        ]["pair"].tolist(),
    }
    (args.output / "q2_persistence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
