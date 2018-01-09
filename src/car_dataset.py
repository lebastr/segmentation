import os

import numpy as np
import PIL.Image as Img
from torch.utils.data.dataset import Dataset


class CarDataset(Dataset):
    def __init__(self, path):
        self.imgs_path = os.path.join(path, 'images_prepped_train')
        self.anns_path = os.path.join(path, 'annotations_prepped_train')

        self.imgs_files = os.listdir(self.imgs_path)

    def __len__(self):
        return len(self.imgs_files)

    def __getitem__(self, i):
        img_path = os.path.join(self.imgs_path, self.imgs_files[i])
        ann_path = os.path.join(self.anns_path, self.imgs_files[i])
        img = Img.open(img_path)
        ann = Img.open(ann_path)

        return np.float32(np.array(img).transpose([2,0,1])), np.int32(np.array(ann))
