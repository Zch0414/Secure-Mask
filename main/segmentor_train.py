import argparse
import logging
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from segmentor_eval import eval_net
from models import UNet
from torch.utils.tensorboard import SummaryWriter
from utils.segmentor_dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = '../CSI-Image data/csi/'#'/Users/lxa/Documents/takeAwayCode&Data/Pytorch-UNet-master/csi/'#
dir_mask_train = './mask_train/'
dir_mask_test = './mask_test/'
dir_checkpoint = './checkpoints/'
dir_mask = './mv2mask/mask_mod/'
dir_save_train = './attackData_train_1/'
dir_save_test = './attackData_test_1/'

def data_output(net, loader, dir, device, show=False):
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

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):

    # dataset_train = BasicDataset(dir_img, dir_mask_train, img_scale)
    # dataset_test = BasicDataset(dir_img, dir_mask_test, img_scale)
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    dataset_train, dataset_test = random_split(dataset, [n_train, n_val])
    n_train = len(dataset_train)
    n_val = len(dataset_test)
    print(n_train, n_val)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='LR_{0}_BS_{1}_SCALE_{2}'.format(lr, batch_size, img_scale))
    global_step = 0

    logging.info('''Starting training:
            Epochs:          {0}
            Batch size:      {1}
            Learning rate:   {2}
            Training size:   {3}
            Validation size: {4}
            Checkpoints:     {5}
            Device:          {6}
            Images scaling:  {7}
        '''.format(epochs, batch_size, lr, n_train, n_val, save_cp, device.type, img_scale))
    weight_p, bias_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer = optim.Adam([{'params': weight_p, 'weight_decay': 1e-5},
                            {'params': bias_p, 'weight_decay': 0}], lr=lr)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc='Epoch {0}/{1}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, "Network has been defined with {0} input channels'format(), \
                          but loaded images have {0} channels. Please check that, \
                          the images are loaded correctly.".format(net.n_channels, imgs.shape[1])
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                imgs = nn.functional.interpolate(imgs, size=[64, 128])
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (batch_size)) == 0:
                    # for tag, value in net.named_parameters():
                    # tag = tag.replace('.', '/')
                    # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device, writer, global_step)

                    train_loader_save = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0,
                                              pin_memory=True, drop_last=True)
                    val_loader_save = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                            pin_memory=True, drop_last=True)
                    data_output(net, val_loader_save, dir_save_test, device, show=True)
                    data_output(net, train_loader_save, dir_save_train, device, show=True)

                    scheduler.step()
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    if net.n_classes == 1:
                        mask = torch.cat((true_masks, torch.sigmoid(masks_pred) > 0.5), dim=2)
                        writer.add_images('masks/train_epoch%i' % global_step, mask, global_step)
                        # writer.add_images('masks/pred_epoch%i'%global_step, torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP_epoch{0}.pth'.format(epoch + 1))
            logging.info('Checkpoint {0} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=40,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {0}'.format(device))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=30, n_classes=1, bilinear=True)
    # net = autoEncoder(n_channels=150, n_classes=1, bilinear=True)
    logging.info('Network:\n'
                 '\t{0} input channels\n'
                 '\t{1} output channels (classes)\n'
                 '\t{2} upscaling'.format(net.n_channels, net.n_classes,
                                          "Bilinear" if net.bilinear else "Transposed conv"))

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {0}'.format(args.load))

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
