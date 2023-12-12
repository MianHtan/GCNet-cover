import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision

import numpy as np
import logging
import os

from pathlib import Path

from PIL import Image


def read_img(filename, resize):
    ext = os.path.splitext(filename)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.jpg' or ext == '.tif':
        img = Image.open(filename)
        img = img.resize(resize)
        img = np.array(img)
    return img

def read_disp(filename, resize):
    disp = Image.open(filename)
    w, h = disp.size
    disp = disp.resize(resize)
    disp = np.array(disp)

    # img has been resized, thus the disparity should resized
    scale = w/resize[0]
    disp = disp / scale

    # generate mask
    disp[np.isnan(disp)] = 999
    valid = disp != 999
    # valid = np.ones_like(disp)
    return disp, valid