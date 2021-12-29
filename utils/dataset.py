import os 
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


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
    def __init__(self, root, transforms, channel):
        self.image_paths = sorted(make_dataset(root))
        self.transforms = transforms
        self.image_channel = channel
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.image_channel == 1:
            image = image.convert('L')
        elif self.image_channel == 3:
            image = image.convert('RGB')
        else:
            raise ValueError("[!]Expect incorrect image format. The channel of image should 1 or 3!")
        return self.transforms(image)

