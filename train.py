import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataset import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F


def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wdice = 1 - (2 * inter + 1)/(union + 1)
    return (wbce + wdice).mean()


def train(train_loader, model, optimizer, epoch, train_save):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescaling the inputs (img/gt) ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1 = model(images)

            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = joint_loss(lateral_map_1, gts)
            loss = loss1 + loss2 + loss3 + loss4

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)

        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-1: {:.4f}, '
                  'lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show()))

    # ---- save model_lung_infection ----
    save_path = './weight/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'nCoVSegNet-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'nCoVSegNet-%d.pth' % (epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=True,
                        help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers in dataloader. In windows, set num_workers=0')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='ResNet50',
                        help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50')
    parser.add_argument('--train_path', type=str,
                        default='/home/ljn/code/nCoVSegNet/data/train')
    parser.add_argument('--train_save', type=str, default=None,
                        help='Use custom save path')

    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)

    if opt.backbone == 'Res2Net50':
        print('Backbone loading: Res2Net50')
        # from module.Res2Net import Res2Net
    elif opt.backbone == 'ResNet50':
        print('Backbone loading: ResNet50')
        from module.nCoVSegNet import nCoVSegNet
    elif opt.backbone == 'VGGNet16':
        print('Backbone loading: VGGNet16')
        # from module.VGGNet import VGGNet
    else:
        raise ValueError('Invalid backbone parameters: {}'.format(opt.backbone))

    model = nCoVSegNet(n_class=opt.n_classes).cuda()

    ## transfer learning
    # model_path = '/home/ljn/code/nCoVSegNet/code/lidc_weight/nCoVSegNet-100.pth'
    # model = nCoVSegNet(n_class=opt.n_classes)
    # state_dict = torch.load(model_path, map_location='cpu')
    # model.load_state_dict(state_dict, strict=False)
    # model.cuda()

    print('Use custom save path')
    train_save = opt.train_save

    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from utils.utils import CalParams
        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
        CalParams(model, x)

    # ---- load training sub-modules ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = len(train_loader)

    # ---- start !! -----
    print("#"*20, "\nStart Training (nCoVSegNet-{})\n{}\n----\n".format(opt.backbone, opt), "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, train_save)