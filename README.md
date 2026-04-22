# Spatio-Temporal SAR-Optical Data Fusion for Cloud Removal via a Deep Hierarchical Model

**Authors: Alessandro Sebastianelli, Erika Puglisi, Maria Pia Del Rosso, Jamila Mifdal, Artur Nowakowki, Fiora Pirri, Pierre Philippe Mathieu and Silvia Liberata Ullo**


**PLEASE BE AWARE THAT THIS REPO IS CURRENTLY UNDER MAINTENANCE, WE ARE UPGRADING THE CODE**
**A NEWER IMPLEMENTATION WILL APPEAR SOON**

## Fork note

This repository is a fork/adaptation of the original PLFM implementation and keeps the same core objective: combining optical time-series and SAR data for cloud removal.

The proposed PLFM model combines a time-series of optical images and a SAR image to remove clouds from optical images.


|Cloudy Image|Model Prediction|Ground Truth|
:-----------:|:-----------:|:-----------:
![](res/cloudy.png) | ![](res/prediction.png) | ![](res/gt.png)


## Installation

Install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

### Windows/Conda install note (rasterio/GDAL)

This project now uses a modern `rasterio` range (`>=1.3.9,<2.0`) to avoid old GDAL build issues on Windows.
In most cases, this is enough:

```
pip install -r requirements.txt
```

If `pip install -r requirements.txt` still fails with a `rasterio`/`GDAL` error, use this fallback:

```
conda create -n plfm python=3.9 -y
conda activate plfm
conda install -c conda-forge rasterio gdal -y
pip install -r requirements.txt
```

Quick verification:

```
python -c "import rasterio, tensorflow as tf; print('rasterio', rasterio.__version__); print('tf', tf.__version__)"
```

If TensorFlow Addons is missing at runtime, install:

```
pip install tensorflow-addons==0.18.0
```

### Compatibility updates in this fork

To make setup and training work reliably on current Windows/Conda environments, this fork includes the following updates.

Code changes:
- removed legacy `keras_contrib` imports from `models/cGAN.py` and `models/dualcGAN.py` (these imports were unused and blocked startup),
- kept model logic unchanged while removing that hard dependency.

Library updates:
- upgraded `rasterio` requirement from `1.2.10` to `>=1.3.9,<2.0` to avoid old GDAL build/DLL issues,
- pinned TensorFlow stack to a known-working pair: `tensorflow==2.10.1` and `tensorflow-addons==0.18.0`,
- added missing runtime dependencies used by `utils/metrics.py`: `scikit-image` and `sewar`.

## Data workflow

### Zone-based dataset preprocessing

Preprocess raw aligned GeoTIFF triplets from `raw_data/clear`, `raw_data/cloudy`, and `raw_data/sar` into the zone format expected by the loader:

```
python preprocess.py --raw_data raw_data_path --output dataset_path --tile_size 256
```

This creates a dataset like:

```
dataset_path/
  zone_A/
    cloudy/
    clear/
    sar/
  zone_B/
  zone_C/
  zone_D/
```

### Naming conventions

#### Raw input naming (`raw_data/clear`, `raw_data/cloudy`, `raw_data/sar`)

The preprocessor matches files by extracting the **last numeric token** from each filename.
All three modalities must share the same final numeric ID.

Examples that match as one triplet:
- `clear_scene_0012.tif`
- `cloudy_scene_0012.tif`
- `sar_scene_0012.tif`

Accepted pattern rule:
- filenames can differ in prefix/text,
- but each file must contain digits,
- and the **last** digit group must be the same across `clear`, `cloudy`, and `sar`.

#### Preprocessed tile naming (output dataset)

For each extracted tile, the script writes the same filename into:
- `dataset_path/zone_A/clear/`
- `dataset_path/zone_A/cloudy/`
- `dataset_path/zone_A/sar/`
(and similarly for `zone_B`, `zone_C`, `zone_D` when `--zone_split quadrants` is used).

Output filename format:
`zoneX_img_<sample_id>_<tile_idx>.tif`

Where:
- `zoneX` is `zoneA`, `zoneB`, `zoneC`, or `zoneD` (without underscore in filename),
- `<sample_id>` is the matched numeric ID from raw filenames,
- `<tile_idx>` is a zero-padded per-zone counter (e.g. `0001`, `0002`, ...).

Example:
- `zoneA_img_0012_0007.tif`

### What preprocessing does automatically

`preprocess.py` is responsible for preparing raw GeoTIFF triplets for training.  
To avoid mismatches and duplicated work, provide aligned raw inputs and let the script handle the transformations below.

The preprocessing code automatically:
- matches `clear`, `cloudy`, and `sar` images by the shared final numeric ID in filenames,
- validates geometry consistency (same width/height and geotransform across each triplet),
- reads optical data as RGB bands and SAR as single-channel input,
- normalizes optical tiles (`minmax` by default, or `std` with `--optical_norm std`),
- converts SAR to dB when needed, clips to `[-25, 10]`, then min-max normalizes,
- splits scenes into fixed `tile_size` patches and assigns each patch to zones (`zone_A`...`zone_D` with quadrant split by default),
- writes processed tiles as float32 GeoTIFFs with consistent naming in each zone folder.

So, before running preprocessing:
- do not manually normalize pixel values,
- do not manually tile the scenes,
- do not manually rename tiles to the final `zoneX_img_*` format,
- do not pre-split into zone folders.

Keep raw data simple: one full-scene GeoTIFF per timestamp in each of `clear/`, `cloudy/`, and `sar/`, with matching numeric IDs.

During training with `python main.py --train dataset_path`, the loader:
- collects shared base filenames in each zone,
- builds cloudy input sequences,
- uses aligned `clear` (target) and `sar` (conditioning) frames from the same base name.

### Validate dataset before training

Run a quick structural validation before training:

```
python validate_dataset.py --dataset dataset_path --sequence_size 3
```

This checks:
- `zone_*` folder presence,
- filename alignment across `cloudy/`, `clear/`, and `sar/`,
- sampled width/height consistency across modalities,
- number of trainable sequences available.

## Usage

### Train

To train PLFM:

```
python main.py --train dataset_path
```

### Predict/Test

After training (or after placing pretrained checkpoints in `weights/`), run test mode to export predictions as GeoTIFFs:

```
python main.py --test dataset_path
```

Optional custom output directory:

```
python main.py --test dataset_path predictions_out
```

Predictions are saved under:

```
predictions_out/
  zone_A/
    pred_<sample_name>.tif
  zone_B/
  ...
```

To change default parameters please look at [models configuration file](models/models_config.py).


## Dataset
The dataset will be available soon.

## Cite our papers

The dataset has been created using our tool proposed in: 

    @article{sebastianelli2021automatic,
        title={Automatic dataset builder for Machine Learning applications to satellite imagery},
        author={Sebastianelli, Alessandro and Del Rosso, Maria Pia and Ullo, Silvia Liberata},
        journal={SoftwareX},
        volume={15},
        pages={100739},
        year={2021},
        publisher={Elsevier}
    }


The PLFM is presented in

    @article{sebastianelli2022clouds,
        author={Sebastianelli, Alessandro and Puglisi, Erika and Del Rosso, Maria Pia and Mifdal, Jamila and Nowakowski, Artur and Mathieu, Pierre Philippe and Pirri, Fiora and Ullo, Silvia Libearata},
        title={Spatio-Temporal SAR-Optical Data Fusion for Cloud Removal via a Deep Hierarchical Model},
        journal={Submitted to IEEE Transactions on Geoscience and Remote Sensing},
        publisher={IEEE},
        note = {arXiv preprint arXiv:2106.12226. https://arxiv.org/abs/2106.12226}
    }
