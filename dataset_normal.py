import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


def convert_to_newtons(forces):
    """Convert raw force readings to Newtons."""
    return (-0.0302 * forces + 29.229)


class MultiDirectoryTactileDataset(Dataset):
    def __init__(self, data_mappings, transform_r=None, transform_g=None, num_frames=None, is_rgb=False):
        """
        Args:
            data_mappings (list of tuples): List containing tuples with
                (img_dir1, img_dir2, force_dir).
            transform_r (callable, optional): Transform for RGB images.
            transform_g (callable, optional): Transform for grayscale images.
            num_frames (int, optional): Number of frames to use. If None, use all available frames.
            is_rgb (bool): Whether the images are RGB or grayscale.
        """
        self.data_mappings = data_mappings
        self.transform = transform_r if is_rgb else transform_g
        self.num_frames = num_frames
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        data = []

        for dir_idx, (img_dir1, img_dir2, force_dir) in enumerate(self.data_mappings):  # Maintain directory order
            img_files1 = sorted([f for f in os.listdir(img_dir1) if f.endswith('.png') or f.endswith('.jpg')])
            img_files2 = sorted([f for f in os.listdir(img_dir2) if f.endswith('.png') or f.endswith('.jpg')])
            force_files = sorted([f for f in os.listdir(force_dir) if f.endswith('.txt')])

            reference_frame_index = 0  # Always using first frame as reference

            reference_img1_path = os.path.join(img_dir1, img_files1[reference_frame_index])
            reference_img2_path = os.path.join(img_dir2, img_files2[reference_frame_index])
            reference_img1 = Image.open(reference_img1_path).convert('RGB')
            reference_img2 = Image.open(reference_img2_path).convert('RGB')

            dir_data = []  # Store data for this directory

            for i in range(reference_frame_index, len(img_files1)):
                img_file1 = img_files1[i]
                img_file2 = img_files2[i]
                force_file = force_files[i]

                try:
                    frame_index = int(force_file.split('_')[2].split('.')[0])  # Extract frame index
                except ValueError:
                    print(f"Warning: Could not extract frame index from {force_file}, skipping.")
                    continue  # Skip if format is incorrect

                force_value = self._extract_sum_from_file(os.path.join(force_dir, force_file))

                if force_value < 0:
                    continue

                dir_data.append((img_dir1, img_dir2, force_dir, img_file1, img_file2, reference_img1, reference_img2,
                                force_value, frame_index))

            # Sort frames *within* this directory tuple
            dir_data.sort(key=lambda x: x[8])  

            # Append to final dataset, maintaining directory order
            data.extend(dir_data)

        # Debugging: Print directory order and frame indices
        for i, d in enumerate(self.data_mappings):
            dir_frames = [entry[8] for entry in data if entry[0] == d[0] and entry[1] == d[1] and entry[2] == d[2]]
            print(f"Directory {i}: {d} -> Frame indices: {dir_frames}")

        # # Keep first `num_frames` while maintaining directory order
        # if self.num_frames and self.num_frames < len(data):
        #     data = data[:self.num_frames]  

        return data



    def _extract_sum_from_file(self, file_path):
        """Extract the force value from a file."""
        with open(file_path, 'r') as file:
            data = file.readline()
            if data:
                force_values = data.split(',')[-1].strip()  # Extract force value
            return convert_to_newtons(float(force_values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dir1, img_dir2, force_dir, img_file1, img_file2, reference_img1, reference_img2, force_value, frame_index = \
            self.data[idx]

        img_path1 = os.path.join(img_dir1, img_file1)
        img_path2 = os.path.join(img_dir2, img_file2)

        # Load current images
        current_img1 = Image.open(img_path1).convert('RGB')
        current_img2 = Image.open(img_path2).convert('RGB')

        # Apply transforms
        if self.transform:
            current_img1 = self.transform(current_img1)
            current_img2 = self.transform(current_img2)
            reference_img1 = self.transform(reference_img1)
            reference_img2 = self.transform(reference_img2)

        return current_img1, reference_img1, current_img2, reference_img2, torch.tensor(force_value, dtype=torch.float32)
