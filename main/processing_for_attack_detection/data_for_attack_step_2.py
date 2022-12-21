import numpy as np
import cv2
import os
from os.path import splitext
from os import listdir

csi_dir = '../CSI-Image data/csi/'
dir_train = './attackData_train/'
dir_test = './attackData_test/'
for dir in listdir(csi_dir):
    if 's1p0' in dir:
        ids = []
        dir_ = os.path.join(csi_dir, dir)
        dirtrain = os.path.join(dir_train, dir)
        if not os.path.exists(dirtrain):
            os.mkdir(dirtrain)
        dirtest = os.path.join(dir_test, dir)
        if not os.path.exists(dirtest):
            os.mkdir(dirtest)
        num = len(listdir(dir_))
        print(num)
        num_test = num//10
        num_train = num - num_test
        ids.extend([os.path.join(dir, splitext(file)[0]) for file in listdir(dir_)  # cisdir/fielname

                         if not file.startswith('.')])
        count = 0

        id = []
        for item in ids:
            filename = item.split('\\')[1]
            id.append(int(filename.split('_')[2]))
            id.sort()

        for index, item in enumerate(ids):
            filename = item.split('\\')[1]
            filename = filename.split('_')[0] + '_' + filename.split('_')[1] + '_' + str(id[index])
            if index < num_train:
                filename = os.path.join(dirtrain, filename)
            else:
                filename = os.path.join(dirtest, filename)
            merge = np.zeros((2*64, 128, 1))
            cv2.imwrite('%s.png' % filename, merge)