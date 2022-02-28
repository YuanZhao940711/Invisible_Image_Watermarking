# encoding: utf-8

import os 
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def make_dataset(dir):
    image_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    for root, _, filename in sorted(os.walk(dir)):
        for fname in filename:
            if is_image_file(fname):
                path = os.path.join(root,fname)
                image_paths.append(path)
    assert len(image_paths) >0, 'Cannot load enough images!'
    return image_paths


class ImageDataset(Dataset):
    def __init__(self, root, transforms):
        self.image_paths = sorted(make_dataset(root))
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transforms(image)