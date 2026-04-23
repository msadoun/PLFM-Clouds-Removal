from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main():
    root = Path("ci_raw_data")
    for sub in ["clear", "cloudy", "sar"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    transform = from_origin(0, 0, 1, 1)
    profile_rgb = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 3,
        "dtype": "float32",
        "transform": transform,
    }
    profile_sar = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
    }

    for i in range(1, 5):
        idx = f"{i:04d}"
        clear = np.random.rand(3, 256, 256).astype("float32")
        cloudy = np.clip(clear + np.random.normal(0, 0.08, clear.shape).astype("float32"), 0, 1)
        sar = np.random.rand(1, 256, 256).astype("float32")

        with rasterio.open(root / "clear" / f"clear_scene_{idx}.tif", "w", **profile_rgb) as dst:
            dst.write(clear)
        with rasterio.open(root / "cloudy" / f"cloudy_scene_{idx}.tif", "w", **profile_rgb) as dst:
            dst.write(cloudy)
        with rasterio.open(root / "sar" / f"sar_scene_{idx}.tif", "w", **profile_sar) as dst:
            dst.write(sar)


if __name__ == "__main__":
    main()
