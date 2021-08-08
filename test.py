# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from module.nCoVSegNet import nCoVSegNet as Network
from utils.dataset import test_dataset
from utils.evaluation import get_DC, get_sensitivity, get_specificity, get_precision, calVOE, calRVD
from imageio import imsave


def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--data_path', type=str, default='/home/ljn/code/nCoVSegNet/data/test',
                        help='Path to test data')
    parser.add_argument('--pth_path', type=str, default='./weight/nCoVSegNet-50.pth',
                        help='Path to weights file')
    parser.add_argument('--save_path', type=str, default='./Results/',
                        help='Path to save the predictions.')
    parser.add_argument('--backbone', type=str, default='ResNet50',
                        help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='choose which GPU device you want to use')
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_device)

    model = Network()
    model.load_state_dict(torch.load(opt.pth_path), strict=False)
    model.cuda()
    model.eval()

    image_dir = '{}/image/'.format(opt.data_path)
    gt_dir = '{}/mask/'.format(opt.data_path)
    dice = 0
    sen = 0
    spe = 0
    pre = 0
    voe = 0
    rvd = 0
    for subindex in os.listdir(image_dir):
        lateral_map = []
        gt_map = []
        image_root = os.path.join(image_dir, str(subindex) + '/')
        gt_root = os.path.join(gt_dir, str(subindex) + '/')
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        save_paths = opt.save_path + str(subindex)
        os.makedirs(save_paths, exist_ok=True)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(image)
            res = lateral_map_1
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            lateral_map.append(res)
            gt_map.append(gt)
            imsave(opt.save_path + str(subindex) + '/' + name, (res*255).astype(np.uint8))
        pred = np.array(lateral_map)
        pred = torch.from_numpy(pred)
        tar = np.array(gt_map)
        tar = torch.from_numpy(tar)
        print(f'patient:', subindex)
        dice_new = get_DC(pred, tar)
        sen_new = get_sensitivity(pred, tar)
        spe_new = get_specificity(pred, tar)
        pre_new = get_precision(pred, tar)
        voe_new = calVOE(pred, tar)
        rvd_new = calRVD(pred, tar)
        dice += dice_new
        sen += sen_new
        spe += spe_new
        pre += pre_new
        voe += voe_new
        rvd += rvd_new
        print(f'new dice:%.4f, new sen:%.4f, new spe:%.4f, new pre:%.4f, new voe:%.4f, new rvd:%.4f' %
              (dice_new, sen_new, spe_new, pre_new, voe_new, rvd_new))
    dice /= len(os.listdir(image_dir))
    sen /= len(os.listdir(image_dir))
    spe /= len(os.listdir(image_dir))
    pre /= len(os.listdir(image_dir))
    voe /= len(os.listdir(image_dir))
    rvd /= len(os.listdir(image_dir))
    print(dice, sen, spe, pre, voe, rvd)
    print('Test Done!')


if __name__ == "__main__":
    inference()