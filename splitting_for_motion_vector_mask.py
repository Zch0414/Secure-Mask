import os
from shutil import copy

dir_total_mv = './mv2mask/mask_mod/'
dir_train_mv = './mask_train/'
if not os.path.exists(dir_train_mv):
    os.mkdir(dir_train_mv)
dir_test_mv = './mask_test/'
if not os.path.exists(dir_test_mv):
    os.mkdir(dir_test_mv)

def split_mv(mask_total, mask_train, mask_test):
    for i in [3, 5, 6, 7, 8, 9, 10]:
        fileName = 's1p2_' + str(i)
        filePath = mask_total + fileName
        n_file = len(os.listdir(filePath))
        n_test = n_file // 10
        n_train = n_file - n_test
        for m in range(10, n_train+10):
            from_path = filePath + '/' + fileName + '_' + str(m) + '.png'
            to_path = mask_train + fileName + '/' 
            if not os.path.exists(to_path):
                os.mkdir(to_path)
            copy(from_path, to_path)
        for k in range(n_train+10, n_file+10):
            from_path = filePath + '/' + fileName + '_' + str(k) + '.png'
            to_path = mask_test + fileName + '/'
            if not os.path.exists(to_path):
                os.mkdir(to_path)
            copy(from_path, to_path)


# In[12]:


split_mv(dir_total_mv, dir_train_mv, dir_test_mv)


# In[ ]:




