import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_loss import dice_coeff


def eval_net(net, loader, device, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    #net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    #with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        imgs = torch.nn.functional.interpolate(imgs, size=[64, 128])
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)
        mask_pred = torch.sigmoid(mask_pred) > 0.5
        mask_pred = mask_pred.float()
        tot += dice_coeff(mask_pred, true_masks).item()
        #pbar.update()
    mask = torch.cat((true_masks, mask_pred), dim=2)
    writer.add_images('masks/test_epoch%i' % global_step, mask, global_step)
    net.train()
    return tot / n_val
