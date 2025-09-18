import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from scipy.signal import savgol_filter

def smooth_force_series(forces, window_length=11, polyorder=2):
    forces = np.array(forces)
    if len(forces) < window_length:
        return forces  # skip smoothing if too few points
    return savgol_filter(forces, window_length=window_length, polyorder=polyorder)


class MultiDirectoryTactileDataset(Dataset):
    def __init__(self, data_mappings, transform_r=None, transform_g=None, num_frames=None, is_rgb=False):
        self.data_mappings = data_mappings
        self.transform = transform_r if is_rgb else transform_g
        self.num_frames = num_frames
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []

        for dir_idx, (img_dir1, img_dir2, force_dir) in enumerate(self.data_mappings):
            img_files1 = sorted([f for f in os.listdir(img_dir1) if f.endswith('.png') or f.endswith('.jpg')])
            img_files2 = sorted([f for f in os.listdir(img_dir2) if f.endswith('.png') or f.endswith('.jpg')])
            force_files = sorted([f for f in os.listdir(force_dir) if f.endswith('.txt')])

            reference_frame_index = 0
            reference_img1_path = os.path.join(img_dir1, img_files1[reference_frame_index])
            reference_img2_path = os.path.join(img_dir2, img_files2[reference_frame_index])
            reference_img1 = Image.open(reference_img1_path).convert('RGB')
            reference_img2 = Image.open(reference_img2_path).convert('RGB')

            dir_data = []
            force_paths = []
            force_values = []

            for i in range(reference_frame_index, len(force_files)):
                force_file = force_files[i]
                try:
                    frame_index = int(force_file.split('_')[2].split('.')[0])
                except ValueError:
                    continue
                force_path = os.path.join(force_dir, force_file)
                force_value = self._extract_force_from_file(force_path)
                force_paths.append((i, frame_index, img_files1[i], img_files2[i], force_file, force_path))
                force_values.append(force_value)

            smoothed_forces = smooth_force_series(force_values, window_length=11, polyorder=2)

            for i, (img_idx, frame_index, img_file1, img_file2, force_file, force_path) in enumerate(force_paths):
                force_value = smoothed_forces[i]

                zero_threshold = 0.3
                keep_prob_for_zero = 0.2

                if abs(force_value) < zero_threshold:
                    if random.random() > keep_prob_for_zero:
                        continue

                dir_data.append((img_dir1, img_dir2, force_dir, img_file1, img_file2,
                                 reference_img1, reference_img2, force_value, frame_index))

            dir_data.sort(key=lambda x: x[8])
            data.extend(dir_data)

        if self.num_frames and self.num_frames < len(data):
            data = random.sample(data, self.num_frames)

        return data

    def _extract_force_from_file(self, file_path):
        with open(file_path, 'r') as file:
            line = file.readline()
            if line:
                fx, fy, fz = map(float, line.strip().split(','))
                return np.sqrt(fx ** 2 + fy ** 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dir1, img_dir2, force_dir, img_file1, img_file2, reference_img1, reference_img2, force_value, frame_index = self.data[idx]

        img_path1 = os.path.join(img_dir1, img_file1)
        img_path2 = os.path.join(img_dir2, img_file2)

        current_img1 = Image.open(img_path1).convert('RGB')
        current_img2 = Image.open(img_path2).convert('RGB')

        if self.transform:
            current_img1 = self.transform(current_img1)
            current_img2 = self.transform(current_img2)
            reference_img1 = self.transform(reference_img1)
            reference_img2 = self.transform(reference_img2)

        return current_img1, reference_img1, current_img2, reference_img2, torch.tensor(force_value, dtype=torch.float32)
