from tensorflow.keras.models import load_model
from models.convLSTM import convLSTM
from models.headPLFM import headPLFM
from models.cGAN import cGAN
from models.models_config import *
from utils.datasetHandler import *
import numpy as np
import rasterio
import os

class PLFM:
    def __init__(self, path):
        self.lstm = convLSTM(len_series=LSTM_SETTINGS['SERIES_SIZE'], img_shape=LSTM_SETTINGS['IMAGE_SHAPE'])
        self.gan = cGAN(img_shape=GAN_SETTINGS['IMAGE_SHAPE'])
        self.head = headPLFM(HEAD_SETTINGS['IMAGE_SHAPE'], self.lstm.model, self.gan.generator)

        # Load pre-trained checkpoints when available.
        lstm_path = os.path.join(path, 'lstm.h5')
        gan_path = os.path.join(path, 'gan.h5')
        gan_d_path = os.path.join(path, 'gan-d.h5')
        head_path = os.path.join(path, 'head.h5')

        if os.path.exists(lstm_path):
            self.lstm.model = load_model(lstm_path)
        if os.path.exists(gan_path):
            self.gan.generator = load_model(gan_path)
        if os.path.exists(gan_d_path):
            self.gan.discriminator = load_model(gan_d_path)
        if os.path.exists(head_path):
            self.head.model = load_model(head_path)

    def train(self, dataset_path):
        # Train from scratch
        self.lstm = convLSTM(len_series = LSTM_SETTINGS['SERIES_SIZE'], img_shape = LSTM_SETTINGS['IMAGE_SHAPE'])
        self.gan = cGAN(img_shape = GAN_SETTINGS['IMAGE_SHAPE'])

        # Default: Load the proposed dataset
        if dataset_path=='SeriesSen1-2':
            s2_paths, s2_zones = get_images_path(dataset_path, 'sen2')
            s2_images, cloud_masks = split_S2_images(s2_paths)
            s2_series = get_time_series(s2_images)
            s1_images, s1_zones = get_images_path(dataset_path, 'sen1')
            s1_series = get_time_series(s1_images)
            head_series = [s2_series[:8], s1_series[:8]]
            gan_loader = image_generatorCycleGAN(s2_series[:8], s1_series[:8], batch_size=GAN_SETTINGS['BATCH SIZE'], normalization='minmax', augment=False)
            gan_steps = max(1, len(s2_series[:8])//GAN_SETTINGS['BATCH SIZE'])
        else:
            triplets = collect_zone_triplets(dataset_path)
            zone_samples = build_zone_sequences(triplets, sequence_size=LSTM_SETTINGS['SERIES_SIZE'])
            if len(zone_samples) == 0:
                raise RuntimeError(
                    f"No zone samples found in {dataset_path}. "
                    f"Ensure each zone_* has matching filenames in cloudy/clear/sar "
                    f"and at least {LSTM_SETTINGS['SERIES_SIZE']} timestamps per zone."
                )
            s2_series = zone_samples
            head_series = zone_samples
            gan_loader = image_generatorCycleGAN(zone_samples, None, batch_size=GAN_SETTINGS['BATCH SIZE'], normalization='minmax', augment=False)
            gan_steps = max(1, len(zone_samples)//GAN_SETTINGS['BATCH SIZE'])

        lstm_train_series = s2_series[:8] if isinstance(s2_series, list) and (len(s2_series) > 8) else s2_series
        head_train_series = head_series[:8] if isinstance(head_series, list) and (len(head_series) > 8) else head_series

        # Print Model Settings
        print('\t @LSTM Settings', LSTM_SETTINGS, '\n')
        self.lstm.train(LSTM_SETTINGS['EPOCHS'], 
               lstm_train_series, # Training
               lstm_train_series, # Validation
               LSTM_SETTINGS['BATCH SIZE'])
        self.lstm.model.save(os.path.join('weights', 'lstm.h5'))

        print('\n\t @GAN Settings', GAN_SETTINGS, '\n')
        self.gan.train(
               GAN_SETTINGS['EPOCHS'],
               gan_loader,
               gan_steps,
               GAN_SETTINGS['BATCH SIZE'])
        self.gan.generator.save(os.path.join('weights', 'gan.h5'))
        self.gan.discriminator.save(os.path.join('weights', 'gan-d.h5'))
        
        print('\n\t @PLFM HEAD Settings', HEAD_SETTINGS, '\n')
        self.head = headPLFM(HEAD_SETTINGS['IMAGE_SHAPE'],  self.lstm.model, self.gan.generator)
        self.head.train(HEAD_SETTINGS['EPOCHS'], 
               head_train_series, # Training
               head_train_series, # Validation
               HEAD_SETTINGS['BATCH SIZE'])
        self.head.model.save(os.path.join('weights', 'head.h5'))

    def _build_head_input(self, sample):
        batch_s2 = np.zeros((1, LSTM_SETTINGS['SERIES_SIZE'], 256, 256, 3), dtype=np.float32)
        for j in range(LSTM_SETTINGS['SERIES_SIZE']):
            s2 = get_s2_image(sample["inputs"][j], normalization='minmax')
            batch_s2[0, j, :s2.shape[0], :s2.shape[1], ...] = s2

        s1 = get_s1_image(sample["sar"], normalization='minmax')
        s1 = (2 * s1) - 1.0
        batch_s1 = np.zeros((1, 256, 256, 3), dtype=np.float32)
        batch_s1[0, :s1.shape[0], :s1.shape[1], 0] = s1[..., 0]
        batch_s1[0, :s1.shape[0], :s1.shape[1], 1] = s1[..., 0]
        batch_s1[0, :s1.shape[0], :s1.shape[1], 2] = s1[..., 0]

        lstm_o = self.lstm.model.predict(batch_s2, verbose=0)
        gan_o = self.gan.generator.predict(batch_s1, verbose=0)

        head_input = np.zeros((1, 256, 256, 6), dtype=np.float32)
        head_input[0, :, :, :3] = lstm_o[0, ...]
        head_input[0, :, :, 3:] = gan_o[0, ...]
        return head_input

    def test(self, dataset_path, output_dir='predictions'):
        triplets = collect_zone_triplets(dataset_path)
        samples = build_zone_sequences(triplets, sequence_size=LSTM_SETTINGS['SERIES_SIZE'])
        if len(samples) == 0:
            raise RuntimeError(
                f"No zone samples found in {dataset_path}. "
                f"Need zone_* with aligned cloudy/clear/sar and at least {LSTM_SETTINGS['SERIES_SIZE']} timestamps."
            )

        output_root = os.path.join(output_dir)
        os.makedirs(output_root, exist_ok=True)

        print(f"[-] Running inference on {len(samples)} samples")
        for idx, sample in enumerate(samples, start=1):
            head_input = self._build_head_input(sample)
            pred = self.head.model.predict(head_input, verbose=0)[0]
            pred = np.clip((pred + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

            zone_output_dir = os.path.join(output_root, sample["zone"])
            os.makedirs(zone_output_dir, exist_ok=True)
            pred_name = f"pred_{sample['name']}.tif"
            pred_path = os.path.join(zone_output_dir, pred_name)

            with rasterio.open(sample["target"]) as src:
                profile = src.profile.copy()
            profile.update(dtype='float32', count=3, compress='lzw')
            with rasterio.open(pred_path, 'w', **profile) as dst:
                dst.write(np.moveaxis(pred, -1, 0))

            if idx % 10 == 0 or idx == len(samples):
                print(f"[-] Saved {idx}/{len(samples)} predictions")



