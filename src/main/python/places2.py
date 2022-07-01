import random
import torch
import os
import numpy as np
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform, use_context):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.use_context = use_context

        if use_context == 'train':
            self.paths = glob('{:s}/*.png'.format(img_root))
            self.mask_paths = glob('{:s}/*.png'.format(mask_root))
            self.N_mask = len(self.mask_paths)
        elif use_context == 'test':
            self.paths = img_root
            self.mask_paths = mask_root
            self.N_mask = 1

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        if self.use_context == 'test':
            mask = Image.open(self.mask_paths)
        else:
            mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        if self.use_context == 'test':
            return gt_img * (1-mask), 1-mask, gt_img
        else:
            return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
