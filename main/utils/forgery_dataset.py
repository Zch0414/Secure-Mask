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
    def __init__(self, masks_dir, j_dir, GOP, scale=1):
        self.masks_dir = masks_dir
        self.j_dir = j_dir
        self.GOP =GOP
        self.scale = scale
        self.ids = []
        self.ids_j = []
        self.label = []
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        for dir in listdir(masks_dir):
            if dir.startswith('.'):
                continue
            dir_ = os.path.join(masks_dir, dir)
            self.ids.extend([os.path.join(dir, splitext(file)[0]) for file in listdir(dir_)

                    if not file.startswith('.')])

        for dir in listdir(j_dir):
            if dir.startswith('.'):
                continue
            dir_ = os.path.join(j_dir, dir)
            self.ids_j.extend([os.path.join(dir, splitext(file)[0]) for file in listdir(dir_)

                    if not file.startswith('.')])
        logging.info('Creating dataset with {0} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        assert len(img_nd.shape) == 3, \
            'Alarm: The size of the mask is wrong'

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans/255
        img_pre = img_trans[:, 0:64, 0:128]
        img_true = img_trans[:, 64:128, 0:128]
        return img_pre, img_true

    def __getitem__(self, i):
        idx_i = self.ids[i]
        mask_file = glob(self.masks_dir + idx_i + '.npy')
        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {0}: {1}'.format(idx_i, mask_file)
        mask_i = np.load(mask_file[0])
        assert mask_i.shape == (128, 128, self.GOP), \
            'The size of the mask is wrong for the ID {0}: {1}'.format(idx_i, mask_file)
        mask_i_pre, mask_i_true = self.preprocess(mask_i)

        if np.random.rand() >= 0.5:
            self.label = 1
            mask = np.concatenate((mask_i_true, mask_i_pre), axis=0)
            assert mask.shape == (2*self.GOP, 64, 128), \
                'The size of the input is wrong for the ID {0}: {1}'.formate(idx_i, mask_file)
        else:
            self.label = 0
            j = np.random.randint(len(self.ids_j))
            idx_j = self.ids_j[j]
            mask_file = glob(self.j_dir + idx_j + '.npy')
            assert len(mask_file) == 1, \
                'Either no mask or multiple masks found for the ID {0}: {1}'.format(idx_j, mask_file)
            mask_j = np.load(mask_file[0])
            assert mask_j.shape == (128, 128, self.GOP), \
                'The size of the mask is wrong for the ID {0}: {1}'.format(idx_j, mask_file)
            mask_j_pre, mask_j_true = self.preprocess(mask_j)
            mask = np.concatenate((mask_i_true, mask_j_pre), axis=0)
            assert mask.shape == (2*self.GOP, 64, 128), \
                'The size of the input is wrong for the ID {0}: {1}'.formate(idx_i, mask_file)

        label = np.array(self.label)

        return {
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'label':torch.from_numpy(label).type(torch.FloatTensor)
        }
