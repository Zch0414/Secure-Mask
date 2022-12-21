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
    def __init__(self, csi_dir, scale=1):
        self.csi_dir = csi_dir
        self.scale = scale
        self.ids = []
        self.label = []
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        for dir in listdir(csi_dir):

            if dir.startswith('.'):
                continue

            dir_ = os.path.join(csi_dir, dir)
            self.ids.extend([os.path.join(dir, splitext(file)[0]) for file in listdir(dir_)

                    if not file.startswith('.')])

        for id in self.ids:
            if 's1p0' in id:
                self.label.append(0)
            else:
                self.label.append(1)

        logging.info('Creating dataset with {0} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        csi_file = glob(self.csi_dir + idx + '.mat')

        assert len(csi_file) == 1, \
            'Either no image or multiple images found for the ID {0}: {1}'.format(idx, csi_file)
        csi = loadmat(csi_file[0])['csi']

        #assert img.size == mask.size, \
        #    'Image and mask {0} should be the same size, but are {1} and {2}'.format(idx, img.size, mask.size)

        csi = self.preprocess(csi)
        label = np.array(self.label[i])
        return {
            'csi': torch.from_numpy(csi).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor)
        }
