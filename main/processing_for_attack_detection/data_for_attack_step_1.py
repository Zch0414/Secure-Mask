import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import sys
sys.path.append('..')
from ..models import UNet
from ..utils.segmentor_dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import cv2
import os

dir = './data/attackData_test/'
# dir = './attackData_train/'

def data_output(net, loader, device, show=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.float
    n_val = len(loader)  # the number of batch
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']
            name = batch['name']
            dirname = name[0].split('\\')[0]
            filename = name[0].split('\\')[1]
            dirname = os.path.join(dir, dirname)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            filename = os.path.join(dirname, filename)
            imgs = nn.functional.interpolate(imgs, size=[64, 128])

            imgs = imgs.to(device=device, dtype=torch.float32)
            # TODO: change the threshold

            with torch.no_grad():
                mask_pred = F.sigmoid(net(imgs))
                mask_pred = mask_pred > 0.5
                mask_pred = mask_pred.float()


            if show:
                merge = torch.zeros((1, 1, 2 * 64, 128))
                merge[:, :, 0:64, 0:128] = mask_pred
                merge[:, :, 64:128, 0:128] = true_masks
                merge = 255 * merge.squeeze(0).cpu().numpy().transpose((1,2,0))
                cv2.imwrite('%s.png' %filename, merge)

            #fake = 255*fake_img.squeeze().cpu().numpy().transpose((1,2,0))
            #fake = cv2.cvtColor(cv2.resize(fake, (256, 128)), cv2.COLOR_RGB2BGR)
            #cv2.imwrite('s1p9/%s.png'%(id[0].replace('/','a')), fake)
            pbar.update()

    net.train()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    dir_img = '../CSI-Image data/csi/'#/Users/lxa/Documents/takeAwayCode&Data/Pytorch-UNet-master/csi/'#
    # dir_mask = './mask_train/'
    dir_mask = './data/mask_test/'
    dir_checkpoint = './CP_epoch40_1.0.pth'
    # dir_checkpoint = './CP_epoch40_1.0.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=150, n_classes=1, bilinear=True).to(device)
    net.load_state_dict(torch.load(dir_checkpoint, map_location=device), strict=True)
    dataset = BasicDataset(dir_img, dir_mask)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
    data_output(net, loader, device, show=False)


if __name__ == '__main__':
    main()