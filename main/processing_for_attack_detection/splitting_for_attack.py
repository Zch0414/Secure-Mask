from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
from PIL import Image
import os

dir_train = './attackData_train_1/'
dir_test = './attackData_test_1/'
save_train = './attack/train/'
if not os.path.exists(save_train):
    os.mkdir(save_train)
save_test = './attack/test/'
if not os.path.exists(save_test):
    os.mkdir(save_test)

def dataSplit(dirname, savename, GOP=7):
    for dir in listdir(dirname):
        save_dir = os.path.join(savename, dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        dir_ = os.path.join(dirname, dir)
        ids = []
        ids.extend([os.path.join(dir, splitext(file)[0]) for file in listdir(dir_)
                   
                    if not file.startswith('.')])
        num = len(ids)
        for index, item in enumerate(ids):
            if index > num-GOP :
                break
            filename = item.split('\\')[1]
            merge = np.zeros((2*64, 128, GOP))  #HWC
            img_file = []
            for i in range(GOP):
                img_file.append(glob(dirname + ids[index + i] + '.png'))
                img_i = Image.open(img_file[i][0])
                img_nd = np.array(img_i)
                merge[:, :, i] = img_nd
            filename = os.path.join(save_dir, filename)
            np.save(filename, merge)

dataSplit(dir_test, save_test, GOP=7)
dataSplit(dir_train, save_train, GOP=7)

