"""Check that every state in the compact simulation bundle has usable inputs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def validate(bundle: Path) -> int:
    manifest = bundle / "gromacs_dcm" / "state_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    errors = []
    names = [row["state"] for row in rows]
    for name, count in Counter(names).items():
        if count != 1:
            errors.append(f"duplicate state: {name} ({count} rows)")

    for row in rows:
        state = row["state"]
        source = bundle / row["source_pdb"]
        if not source.is_file():
            errors.append(f"{state}: missing {row['source_pdb']}")
        try:
            vinyls = json.loads(row["vinyl_mapping_json"])
            pairs = json.loads(row["pair_schema_json"])
            if len(vinyls) != int(row["active_vinyl_count"]):
                errors.append(f"{state}: active_vinyl_count mismatch")
            if len(pairs) != int(row["active_pair_count"]):
                errors.append(f"{state}: active_pair_count mismatch")
            for pair in pairs:
                for side in ("left", "right"):
                    if f"V{pair[side]}" not in vinyls:
                        errors.append(f"{state}: pair refers to absent V{pair[side]}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{state}: invalid manifest data: {exc}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    counts = Counter(row["system"] for row in rows)
    print(f"Validated {len(rows)} states across {len(counts)} systems")
    for system, count in sorted(counts.items()):
        print(f"  {system}: {count}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "gromacs_dcm_server_bundle",
        help="Path containing gromacs_dcm/state_manifest.csv and structures/",
    )
    args = parser.parse_args()
    raise SystemExit(validate(args.root))
