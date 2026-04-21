import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def parse_args():
    parser = argparse.ArgumentParser(description="Build zone-based PLFM dataset from raw GeoTIFFs.")
    parser.add_argument("--raw_data", required=True, help="Path to raw data root with clear/, cloudy/, sar/.")
    parser.add_argument("--output", required=True, help="Output dataset path.")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size in pixels (default: 256).")
    parser.add_argument(
        "--optical_norm",
        choices=["minmax", "std"],
        default="minmax",
        help="Optical normalization strategy for cloudy/clear tiles.",
    )
    parser.add_argument(
        "--zone_split",
        choices=["quadrants", "none"],
        default="quadrants",
        help="How to split a scene into zones when zone bboxes are not provided.",
    )
    return parser.parse_args()


def extract_numeric_id(filename):
    matches = re.findall(r"(\d+)", filename)
    return matches[-1] if matches else None


def safe_minmax(arr):
    arr = arr.astype(np.float32)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def safe_std(arr):
    arr = arr.astype(np.float32)
    sd = np.nanstd(arr)
    if sd < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - np.nanmean(arr)) / sd


def normalize_optical(arr, mode):
    if mode == "std":
        arr = safe_std(arr)
        # keep expected range for training code
        return np.clip(arr, 0.0, 1.0)
    return np.clip(safe_minmax(arr), 0.0, 1.0)


def normalize_sar(arr):
    arr = arr.astype(np.float32)
    if np.nanmax(arr) > 1.0:
        arr = 10.0 * np.log10(np.clip(arr, 1e-6, None))
    arr = np.clip(arr, -25.0, 10.0)
    return np.clip(safe_minmax(arr), 0.0, 1.0)


def read_rgb(src, window):
    rgb = src.read([1, 2, 3], window=window).astype(np.float32)
    return np.moveaxis(rgb, 0, -1)


def read_sar(src, window):
    sar = src.read(1, window=window).astype(np.float32)
    return sar[..., np.newaxis]


def zone_name_for_window(x, y, width, height, mode):
    if mode == "none":
        return "zone_A"
    x_mid = width // 2
    y_mid = height // 2
    if y < y_mid and x < x_mid:
        return "zone_A"
    if y < y_mid and x >= x_mid:
        return "zone_B"
    if y >= y_mid and x < x_mid:
        return "zone_C"
    return "zone_D"


def write_tile(dst_path, arr, profile):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    if arr.ndim == 3:
        bands = arr.shape[-1]
        prof.update(dtype="float32", count=bands, compress="lzw")
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(np.moveaxis(arr.astype(np.float32), -1, 0))
    else:
        prof.update(dtype="float32", count=1, compress="lzw")
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)


def collect_raw_triplets(raw_root):
    clear_dir = Path(raw_root) / "clear"
    cloudy_dir = Path(raw_root) / "cloudy"
    sar_dir = Path(raw_root) / "sar"

    def map_ids(folder):
        mapping = {}
        for tif in sorted(folder.glob("*.tif")):
            idx = extract_numeric_id(tif.name)
            if idx is not None:
                mapping[idx] = tif
        return mapping

    clear_map = map_ids(clear_dir)
    cloudy_map = map_ids(cloudy_dir)
    sar_map = map_ids(sar_dir)
    shared_ids = sorted(set(clear_map) & set(cloudy_map) & set(sar_map))
    return [(idx, clear_map[idx], cloudy_map[idx], sar_map[idx]) for idx in shared_ids]


def preprocess(raw_data, output, tile_size, optical_norm, zone_split):
    output = Path(output)
    triplets = collect_raw_triplets(raw_data)
    if not triplets:
        raise RuntimeError("No matching clear/cloudy/sar IDs found in raw data.")

    tile_counter = {}

    for sample_id, clear_fp, cloudy_fp, sar_fp in triplets:
        with rasterio.open(clear_fp) as clear_src, rasterio.open(cloudy_fp) as cloudy_src, rasterio.open(sar_fp) as sar_src:
            if not (clear_src.width == cloudy_src.width == sar_src.width and clear_src.height == cloudy_src.height == sar_src.height):
                raise ValueError(f"Shape mismatch in sample {sample_id}.")
            if not (clear_src.transform == cloudy_src.transform == sar_src.transform):
                raise ValueError(f"Geotransform mismatch in sample {sample_id}.")

            width = clear_src.width
            height = clear_src.height

            for y in range(0, height - tile_size + 1, tile_size):
                for x in range(0, width - tile_size + 1, tile_size):
                    zone = zone_name_for_window(x, y, width, height, zone_split)
                    tile_counter.setdefault(zone, 0)
                    tile_counter[zone] += 1
                    tile_idx = tile_counter[zone]

                    window = Window(col_off=x, row_off=y, width=tile_size, height=tile_size)
                    profile = clear_src.profile.copy()
                    transform = rasterio.windows.transform(window, clear_src.transform)
                    profile.update(height=tile_size, width=tile_size, transform=transform)

                    clear_tile = normalize_optical(read_rgb(clear_src, window), optical_norm)
                    cloudy_tile = normalize_optical(read_rgb(cloudy_src, window), optical_norm)
                    sar_tile = normalize_sar(read_sar(sar_src, window))

                    name = f"{zone.replace('_', '')}_img_{sample_id}_{tile_idx:04d}.tif"
                    write_tile(output / zone / "clear" / name, clear_tile, profile)
                    write_tile(output / zone / "cloudy" / name, cloudy_tile, profile)
                    write_tile(output / zone / "sar" / name, sar_tile, profile)


def main():
    args = parse_args()
    preprocess(
        raw_data=args.raw_data,
        output=args.output,
        tile_size=args.tile_size,
        optical_norm=args.optical_norm,
        zone_split=args.zone_split,
    )


if __name__ == "__main__":
    main()
