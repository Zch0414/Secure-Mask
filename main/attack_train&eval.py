import torch
from torch import nn,optim
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("..")
import numpy as np
import logging
from utils.forgery_dataset import BasicDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
import os
import matplotlib.pyplot as plt
from models import ResNet, Residual

def train_net(net,
              device,
              epochs=20,
              batch_size=32,
              lr=0.001,
              save_cp=True,
              img_scale=1):
    dataset_train = BasicDataset(dir_train, dir_train_j, GOP=7, scale=img_scale)
    dataset_val = BasicDataset(dir_val, dir_val_j, GOP=7, scale=img_scale)
    n_train = len(dataset_train)
    n_val = len(dataset_val)
    print(n_train, n_val)
    train_loader = DataLoader(dataset_train, 
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset_val,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)
    global_step = 0
    logging.info('''Starting training:
    Epochs:            {0}
    Batch size         {1}
    Learning rate      {2}
    Training size      {3}
    Validation size:   {4}
    Checkpoints        {5}
    Devices:           {6}
    Images scaling     {7}
    '''.format(epochs, batch_size, lr, n_train, n_val, save_cp, device, img_scale))

    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    weight_decay = 2e-5
    optimizer = optim.Adam([{'params': weight_p, 'weight_decay':weight_decay},
                      {'params': bias_p, 'weight_decay':0}], lr=lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch{0}/{1}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
                label = batch['label'].to(device=device, dtype=torch.float).unsqueeze(dim=1)
                mask = batch['mask'].to(device=device, dtype=torch.float32)
                
                output = net(mask)
                loss = criterion(output, label)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                pbar.update(mask.shape[0])
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:
                    val_acc, fpr_val, tpr_val, threshold_val = eval_net(net, val_loader, device)
                    train_acc, fpr_train, tpr_train, threshold_train = eval_net(net, train_loader, device)
                    scheduler.step()
                    
                    logging.info('''Epoch result:
                    acc_train: {0}
                    acc_val: {1}'''.format(train_acc, val_acc))
                    
                    save(dir_result_test + 'fpr\\' + 'epoch' + str(epoch+1) + '_' + str(lr) + '_' + str(weight_decay), fpr_val)
                    save(dir_result_test + 'tpr\\' + 'epoch' + str(epoch+1) + '_' + str(lr) + '_' + str(weight_decay), tpr_val)
                    save(dir_result_test + 'threshold\\' + 'epoch' + str(epoch+1) + '_' + str(lr) + '_' + str(weight_decay), threshold_val)
                    
                    plot(fpr_train, tpr_train)
                    plot(fpr_val,tpr_val)
                    
        if save_cp:
            try:
                os.mkdir(dir_save)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_save + 'CP_epoch{0}.pth'.format(epoch+1))
            logging.info('Checkpoint {0} saved !'.format(epoch+1))
                
    return threshold_train, fpr_train, tpr_train, threshold_val, fpr_val, tpr_val

def eval_net(net, loader, device):
    prediction = []
    true = []
    score = []
    n_val = len(loader)
    net.eval()
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            mask, label = batch['mask'], batch['label']
            mask = mask.to(device=device, dtype=torch.float32)
            label = label.to(device=device)
            with torch.no_grad():
                output = net(mask)
            output = torch.sigmoid(output)
            output = output.cpu().numpy()
            output = output.reshape(output.shape[0])
            score.extend(list(output))
            prediction.extend(list((output>0.5).astype(int)))
            true.extend(list(label.cpu().numpy().astype(int)))
            
            pbar.update()
            
    fpr, tpr, thresholds = roc_curve(np.array(true), np.array(score), pos_label=1)
    acc = accuracy_score(prediction, true)

    
    net.train()
    
    return acc, fpr, tpr, thresholds

def plot(fpr, tpr):
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color='darkblue')
    plt.plot(fpr, fpr, color='mediumseagreen', linestyle='--')
    plt.legend(['ROC curve', 'Reference line'])
    plt.text(0.2, 0.7, 'AUROC=%.6f'%auc(fpr, tpr), fontsize=15)
    #plt.axis([0,1,0,1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.show()

def save(path, data):
    data = np.array(data)
    np.save(path, data)

dir_train = './data/Ablationdata/train/'
dir_train_j = './data/Ablationdata/train_j/'
dir_val = './data/Ablationdata/test/'
dir_val_j = './data/Ablationdata/test_j/'
dir_save = './model/'
dir_result_train = './result_train/'
dir_result_test = './result_test/'

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {0}'.format(device))
    
    epochs = 20
    batch_size = 32
    lr = 0.001
    save_cp = True
    img_scale = 1
    
    net = ResNet(Residual)
    net.to(device=device)
    threshold_train, fpr_train, tpr_train, threshold_val, fpr_val, tpr_val = train_net(net=net, 
              device=device, 
              epochs=epochs,
              batch_size=batch_size,
              lr=lr,
              save_cp=save_cp,
              img_scale=img_scale)

    


# In[ ]:


# testdir = '../mask_train/s1p1_10/s1p1_10_15.png'
# testfile = glob(testdir)
# print(len(testfile))
# img = Image.open(testfile[0])
# print(img.size)


# In[ ]:


# for batch in train_loader:
#     mask = batch['mask']
#     label = batch['label']
#     print(mask.shape, label)


# In[ ]:


# len(fpr_train)
# test = score_val
# test.sort()


# In[ ]:


# plt.plot(range(len(score_val)),score_val)
# plt.xlabel('None')
# plt.ylabel('Score')
# plt.grid()
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




