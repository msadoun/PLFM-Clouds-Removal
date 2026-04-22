import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

import rasterio


def natural_key(name):
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return key


def list_tif_stems(folder: Path):
    if not folder.exists():
        return []
    return [p.stem for p in sorted(folder.glob("*.tif"))]


def validate_zone(zone_path: Path, sequence_size: int):
    cloudy = set(list_tif_stems(zone_path / "cloudy"))
    clear = set(list_tif_stems(zone_path / "clear"))
    sar = set(list_tif_stems(zone_path / "sar"))

    shared = sorted(cloudy & clear & sar, key=natural_key)
    report = {
        "zone": zone_path.name,
        "cloudy_count": len(cloudy),
        "clear_count": len(clear),
        "sar_count": len(sar),
        "shared_count": len(shared),
        "missing_in_clear": sorted(cloudy - clear)[:5],
        "missing_in_sar": sorted(cloudy - sar)[:5],
        "missing_in_cloudy": sorted((clear | sar) - cloudy)[:5],
        "sequence_count": max(0, len(shared) - sequence_size + 1),
        "shape_consistent": True,
        "bad_shape_examples": [],
    }

    if not shared:
        report["shape_consistent"] = False
        return report

    for name in shared[: min(len(shared), 20)]:
        paths = [
            zone_path / "cloudy" / f"{name}.tif",
            zone_path / "clear" / f"{name}.tif",
            zone_path / "sar" / f"{name}.tif",
        ]
        shapes = []
        for p in paths:
            with rasterio.open(p) as src:
                shapes.append((src.width, src.height))
        if len(set(shapes)) != 1:
            report["shape_consistent"] = False
            report["bad_shape_examples"].append({"name": name, "shapes": shapes})

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate zone-based PLFM dataset.")
    parser.add_argument("--dataset", required=True, help="Path to dataset root containing zone_* folders.")
    parser.add_argument("--sequence_size", type=int, default=3, help="Sequence length required by training.")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"[!] Dataset path not found: {root}")
        sys.exit(1)

    zones = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("zone_")])
    if not zones:
        print(f"[!] No zone_* directories found in {root}")
        sys.exit(1)

    reports = [validate_zone(z, args.sequence_size) for z in zones]
    total_sequences = sum(r["sequence_count"] for r in reports)

    print("=== Dataset validation report ===")
    print(f"Dataset: {root}")
    print(f"Zones found: {len(zones)}")
    print(f"Required sequence size: {args.sequence_size}")
    print()

    has_errors = False
    for r in reports:
        print(f"[{r['zone']}] cloudy={r['cloudy_count']} clear={r['clear_count']} sar={r['sar_count']} shared={r['shared_count']} sequences={r['sequence_count']}")
        if r["shared_count"] == 0:
            has_errors = True
            print("  - ERROR: no shared filenames across cloudy/clear/sar")
        if r["missing_in_clear"] or r["missing_in_sar"] or r["missing_in_cloudy"]:
            has_errors = True
            print("  - WARN: unmatched filenames found between modalities")
        if not r["shape_consistent"]:
            has_errors = True
            print("  - ERROR: shape mismatch detected in sampled triplets")

    print()
    print(f"Total trainable sequences: {total_sequences}")
    if total_sequences == 0:
        has_errors = True
        print("[!] ERROR: no trainable sequences. Add more timestamps per zone.")

    if has_errors:
        print("[!] Validation finished with issues.")
        sys.exit(2)
    print("[+] Validation passed.")


if __name__ == "__main__":
    main()
