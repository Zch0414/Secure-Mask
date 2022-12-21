from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from scipy.io import loadmat
import os

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir    # Images means csi actually
        self.masks_dir = masks_dir
        self.scale = scale
        self.ids = []
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        for dir in listdir(masks_dir):
            if dir.startswith('.') or 's1p0' in dir:    # Could be wrong, try s1p0
                continue
            dir_ = os.path.join(masks_dir, dir)
            self.ids.extend([os.path.join(dir, splitext(file)[0]) for file in listdir(dir_)

                    if not file.startswith('.')])

        logging.info('Creating dataset with {0} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, name):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if name == 'mask':
            img_trans = img_trans/255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.png')
        img_file = glob(self.imgs_dir + idx + '.mat')
        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {0}: {1}'.format(idx, mask_file)
        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {0}: {1}'.format(idx, img_file)
        mask = Image.open(mask_file[0])
        if mask.size != (128, 64):
            mask = mask.resize((128, 64))
        #mask = mask.convert('1') # convert image to black and white
        img = loadmat(img_file[0])['csi']
        # img = img[:, :, 0:30] #Ablation study
        #assert img.size == mask.size, \
        #    'Image and mask {0} should be the same size, but are {1} and {2}'.format(idx, img.size, mask.size)

        img = self.preprocess(img, name='img')
        mask = self.preprocess(mask, name='mask')
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }