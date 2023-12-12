import torch
import torch.utils.data as data

import numpy as np
import logging
import os

from pathlib import Path
from glob import glob

from .read_data import *


class StereoDataset(data.Dataset):
    def __init__(self, resize):    

        self.is_test = False
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.resize = resize

    def __getitem__(self, index):

        index = index % len(self.image_list)
        
        img1 = read_img(self.image_list[index][0], self.resize)
        img2 = read_img(self.image_list[index][1], self.resize)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        if self.is_test:
            return img1, img2
        disp, valid = read_disp(self.disparity_list[index], self.resize)
        disp = torch.from_numpy(disp).float()
        valid = torch.from_numpy(valid).int()

        return img1, img2, disp, valid
    
    def __len__(self):
        return len(self.image_list)
    
    
class DFC2019(StereoDataset):
    def __init__(self, root, resize, image_set='training'):
        super().__init__(resize)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Track2-RGB-*/*LEFT_RGB.tif')))
        image2_list = sorted(glob(os.path.join(root, 'Track2-RGB-*/*RIGHT_RGB.tif')))
        disp_list = sorted(glob(os.path.join(root, 'Track2-RGB-*/*LEFT_AGL.tif')))
        if image_set == "testing":
            image1_list = image1_list[::20]
            image2_list = image2_list[::20]
            disp_list = disp_list[::20]
        if image_set == "training":
            image1_list_test = image1_list[::20]
            image2_list_test = image2_list[::20]
            disp_list_test = disp_list[::20]
            image1_list = sorted(list(set(image1_list).difference(set(list(image1_list_test)))))
            image2_list = sorted(list(set(image2_list).difference(set(list(image2_list_test)))))
            disp_list = sorted(list(set(disp_list).difference(set(list(disp_list_test)))))
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

def fetch_dataset(dataset_name, root, batch_size, resize, mode="training"):
    if dataset_name == 'DFC2019':
        if mode == 'training':
            dataset = DFC2019(root=root, resize = resize,
                            image_set='training')
        elif mode == 'testing':
            dataset = DFC2019(root=root, resize = resize,
                            image_set='testing')
    else: 
        print("no such a dataset")
    train_loader = data.DataLoader(dataset = dataset, batch_size=batch_size,
                                   pin_memory=True, shuffle=True,
                                   num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, drop_last=True)
    return train_loader