import time
import torch
from torch import nn,optim
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("..")
from utils.detector_dataset import BasicDataset
from models import Detector
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from tqdm import tqdm

dir_csi = '../../CSI-Image data/csi/'
modelpara_path = './model/detector_parameter.pkl'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
TRAIN_FLAG = True
EPOCHS = 15
LR = 1e-5
TEST_PERCENT = 0.1
EVAL_PERCENT = 0.1
BATCH_SIZE = 1
dataset = BasicDataset(dir_csi)
n_test = int(len(dataset) * TEST_PERCENT)
n_eval = int(len(dataset) * EVAL_PERCENT)
n_train = len(dataset) - n_test - n_eval
traindata, testdata, evaldata = random_split(dataset, [n_train, n_test, n_eval])
train_loader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
test_loader = DataLoader(testdata, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
eval_loader = DataLoader(evaldata, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)
print(len(traindata), len(testdata), len(evaldata))
global_step = 0
net = Detector()
net = net.to(device)
optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max' if net.n_classes == 1 else 'min', patience=10)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

def train_and_test(model, criterion, optimizer, scheduler, trainloader, testloader, epochs=10, log_interval=n_train // BATCH_SIZE):
    
    train_acc = list()
    train_acc.append(0)
    test_acc=list()
    test_acc.append(0)
    print('----- Train Start -----')
    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(trainloader):
            batch_x = batch['csi'].to(device=device, dtype=torch.float32)
            batch_y = batch['label'].to(device=device, dtype=torch.float).unsqueeze(dim=1)
            
            assert batch_x.shape[1]/5 == model.n_channel, "Network has been defined with {0} input channels'\
                      but loaded images have {1} channels. Please check that, \
                      the images are loaded correctly.".format(model.n_channel, batch_x.shape[1])
            
            output = model(batch_x)
            optimizer.zero_grad()
            loss = criterion(output, batch_y)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            running_loss += loss.item()
            if (step+1) % log_interval == 0:
                print('The %dth epoch loss: %.4f' %
                      (epoch + 1, running_loss / log_interval))
                running_loss = 0.0
            
        scheduler.step()

        prediction = []
        true = []

        with torch.no_grad():
            for batch in trainloader:
                batch_x = batch['csi'].to(device=device, dtype=torch.float32)
                batch_y = batch['label'].to(device=device)
                output = model(batch_x)
                output = output.cpu().numpy()
                output = output.reshape(output.shape[0])
                prediction.extend(list((output>0.5).astype(int)))
                true.extend(list(batch_y.cpu().numpy().astype(int)))
                
        accuracy = accuracy_score(prediction, true)   
        train_acc.append(accuracy)
        print('Accuracy of the %dth epoch in traning is: %.4f %%' %(epoch+1, accuracy*100))
        
        prediction = []
        true = []
        
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch['csi'].to(device=device, dtype=torch.float32)
                batch_y = batch['label'].to(device=device, dtype=torch.float)
                output = model(batch_x)
                output = output.cpu().numpy()
                output = output.reshape(output.shape[0])
                prediction.extend(list((output>0.5).astype(int)))
                true.extend(list(batch_y.cpu().numpy().astype(int)))              

        accuracy = accuracy_score(prediction, true)  
        test_acc.append(accuracy)
        print('Accuracy of the %dth epoch in testing is: %.4f %%' %(epoch+1, accuracy*100))
        
        model.train()
        
    print('----- Train Finished -----')
    torch.save(model.state_dict(), modelpara_path)
    
    return train_acc, test_acc

def evaluate(model, testloader):
    eval_acc=list()
    eval_acc.append(0)
    model.eval()
    print('------ Evaluate Start -----')

    prediction = []
    true = []
    score = []
    
    n_val = len(testloader)
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave= False) as pbar:
        with torch.no_grad():
            for batch in testloader:
                batch_x = batch['csi'].to(device=device, dtype=torch.float32)
                batch_y = batch['label'].to(device=device, dtype=torch.float)
                output = model(batch_x)
                output = output.cpu().numpy()
                output = output.reshape(output.shape[0])
                score.extend(list(output))
                prediction.extend(list((output>0.5).astype(int)))
                true.extend(list(batch_y.cpu().numpy().astype(int)))
                pbar.update()
        fpr, tpr, thresholds = roc_curve(np.array(true), np.array(score), pos_label=1)
    
        accuracy = accuracy_score(prediction, true)  
        eval_acc.append(accuracy)
    print('Accuracy evaluating is: %.4f %%' % (accuracy*100))
    
    return accuracy, fpr, tpr

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

if __name__ == '__main__':
    if not TRAIN_FLAG:
        train_acc, test_acc = train_and_test(net, criterion, optimizer, scheduler, train_loader, test_loader, epochs=EPOCHS)
        accuracy = evaluate(net, eval_loader)
    else:
        net.load_state_dict((torch.load(modelpara_path)))
        net.eval()
        start = time.time()
        accuracy, fpr, tpr = evaluate(net, eval_loader)
        end = time.time()
        FPS = n_eval / (end - start)
        print(FPS)
        plot(fpr, tpr)