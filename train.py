import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys
import os
# from tqdm import tqdm
import os.path as osp
# from networks.unet2D import UNet
from networks.efficient import EfficientSegBackbone
from dataset.MyoPSDataset import MyoPSDataSet, MyoPSDataSetVal

import random
import timeit
from tensorboardX import SummaryWriter
from loss_functions import loss
from utils.ParaFlop import print_model_parm_nums, print_model_parm_flops, torch_summarize_df

from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model

start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    # Basic
    parser = argparse.ArgumentParser(description="W-like Network for 3D Medical Image Segmentation.")

    # paths
    parser.add_argument("--data_dir", type=str, default='dataset/')
    parser.add_argument("--train_list", type=str, default='list/MyoPS2020/t1.txt')
    parser.add_argument("--val_list", type=str, default='list/MyoPS2020/t2.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/MyoPS2020/baseline/fold5/')
    parser.add_argument("--restore_from", type=str, default='snapshots/MyoPS2020/baseline/fold1/xx.pth')

    # training details
    parser.add_argument("--input_size", type=str, default='288,288')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=5+1)
    parser.add_argument("--compound_coef", type=int, default=0) # B0-B7
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--weight_std", type=bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--not_restore_last", action="store_true")
    parser.add_argument("--save_num_images", type=int, default=2)

    # data aug.
    parser.add_argument("--random_mirror", type=bool, default=True, )
    parser.add_argument("--random_scale", type=bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)

    # others
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--recurrence", type=int, default=1)
    parser.add_argument("--ft", type=bool, default=False)
    parser.add_argument("--ohem", type=str2bool, default='False')
    parser.add_argument("--ohem_thres", type=float, default=0.6)
    parser.add_argument("--ohem_keep", type=int, default=200000)
    return parser


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.Dice = loss.DiceLoss()
        self.Surf = loss.SurfaceLoss(idc=[1,2,3,4,5])
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, distance_map):
        prediction = self.model(imgs)
        if self.model.training ==True:
            Dice_loss = self.Dice(prediction, annotations)
            Surf_loss = self.Surf(prediction, distance_map)
            return Dice_loss, Surf_loss
        else:
            return prediction

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def adjust_alpha(i_iter, num_stemps):
    alpha_begin = 1
    alpha_end = 0.01
    decay = (alpha_begin - alpha_end)/num_stemps
    alpha = alpha_begin - decay * i_iter
    return alpha

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    # num = torch.sum(torch.mul(predict, target), dim=1) + 1
    # den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + 1
    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result



def compute_dice_score(preds, labels):
    # preds: 1x5x3x256x256
    # labels: 1x5x3x256x256

    if isinstance(preds, list):
        shape = np.array(preds[0].shape)
        shape[0] = labels.shape[0]
        shape = tuple(shape)
        predss = torch.zeros(shape).cuda()
        bs_singlegpu = preds[0].shape[0]
        for i in range(len(preds)):
            predss[i * bs_singlegpu:(i + 1) * bs_singlegpu] = preds[i]

        # predss = F.softmax(predss,1)

        pred_LV = predss[:, 1, :, :, :]
        pred_RV = predss[:, 2, :, :, :]
        pred_MY = predss[:, 3, :, :, :]
        pred_edema = predss[:, 4, :, :, :]
        pred_scars = predss[:, 5, :, :, :]

        label_LV = labels[:, 1, :, :, :]
        label_RV = labels[:, 2, :, :, :]
        label_MY = labels[:, 3, :, :, :]
        label_edema = labels[:, 4, :, :, :]
        label_scars = labels[:, 5, :, :, :]

        dice_LV = dice_score(pred_LV, label_LV).cpu().data.numpy()
        dice_RV = dice_score(pred_RV, label_RV).cpu().data.numpy()
        dice_MY = dice_score(pred_MY, label_MY).cpu().data.numpy()
        dice_edema = dice_score(pred_edema, label_edema).cpu().data.numpy()
        dice_scars = dice_score(pred_scars, label_scars).cpu().data.numpy()

    else:

        # preds = preds.cpu().data.numpy()
        # labels = labels.cpu().data.numpy()

        # preds = F.softmax(preds,1)

        pred_LV = preds[:, 1]
        pred_RV = preds[:, 2]
        pred_MY = preds[:, 3]
        pred_edema = preds[:, 4]
        pred_scars = preds[:, 5]

        label_LV = labels[:, 1]
        label_RV = labels[:, 2]
        label_MY = labels[:, 3]
        label_edema = labels[:, 4]
        label_scars = labels[:, 5]

        dice_LV = dice_score(pred_LV, label_LV).cpu().data.numpy()
        dice_RV = dice_score(pred_RV, label_RV).cpu().data.numpy()
        dice_MY = dice_score(pred_MY, label_MY).cpu().data.numpy()
        dice_edema = dice_score(pred_edema, label_edema).cpu().data.numpy()
        dice_scars = dice_score(pred_scars, label_scars).cpu().data.numpy()
    return dice_LV, dice_RV, dice_MY, dice_edema, dice_scars


# for 2D
def predict_sliding(args, net, image, tile_size, num_class):  # image: 1,3,5,256,256, tile_size:256x256
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[3] - tile_size[0]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[1]) / strideHW) + 1)
    # print("Need %i x %i x %i prediction tiles @ stride %i x %i px" % (tile_deps, tile_cols, tile_rows, strideD, strideHW))
    full_probs = np.zeros((image_size[0], num_class, image_size[2], image_size[3], image_size[4])).astype(
        np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros((image_size[0], num_class, image_size[2], image_size[3], image_size[4])).astype(
        np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()
    tile_counter = 0

    for dep in range(image_size[2]):
        for row in range(tile_rows):
            for col in range(tile_cols):
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                x2 = min(x1 + tile_size[1], image_size[4])
                y2 = min(y1 + tile_size[0], image_size[3])
                x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows
                d1 = dep
                d2 = dep+1

                img = torch.squeeze(image[:, :, d1:d2, y1:y2, x1:x2], 2)

                tile_counter += 1
                prediction = net(img)
                prediction = F.softmax(prediction, 1)
                prediction = torch.unsqueeze(prediction, 2)
                # prediction = F.softmax(prediction[0], 1)
                # prediction = torch.unsqueeze(prediction, 2)

                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def validate(args, input_size, model, ValLoader, num_class, loss_seg_Dice, loss_seg_surface, engine):
    # start to validate
    model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.track_running_stats = False

    val_loss = 0.0
    val_LV = 0.0
    val_RV = 0.0
    val_MY = 0.0
    val_emeda = 0.0
    val_scars = 0.0

    for index, batch in enumerate(ValLoader):
        image, label, _, name, _ = batch
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            # pred = predict_multiscale(model, image, (128, 240,240), 4, True, args.recurrence)
            pred = predict_sliding(args, model, image, input_size, num_class)

            # loss = loss_seg_Dice.forward(pred, label) + loss_seg_surface.forward(pred, label)

            dice_LV, dice_RV, dice_MY, dice_emeda, dice_scars = compute_dice_score(pred, label)

            loss = np.array(2-dice_emeda-dice_scars)

            print('%d processd, name: %s, LV:%.4f, RV:%.4f, MY:%.4f, edema:%.4f, scars:%.4f, ' % (index, name, dice_LV, dice_RV, dice_MY, dice_emeda, dice_scars))
            val_loss += torch.Tensor(loss).cuda()
            val_LV += torch.Tensor(dice_LV).cuda()
            val_RV += torch.Tensor(dice_RV).cuda()
            val_MY += torch.Tensor(dice_MY).cuda()
            val_emeda += torch.Tensor(dice_emeda).cuda()
            val_scars += torch.Tensor(dice_scars).cuda()
    val_loss = engine.all_reduce_tensor(val_loss)
    val_LV = engine.all_reduce_tensor(val_LV)
    val_RV = engine.all_reduce_tensor(val_RV)
    val_MY = engine.all_reduce_tensor(val_MY)
    val_emeda = engine.all_reduce_tensor(val_emeda)
    val_scars = engine.all_reduce_tensor(val_scars)

    model.train()
    return val_loss / (index + 1), val_LV / (index + 1), val_RV / (index + 1), val_MY / (index + 1), val_emeda / (index + 1), val_scars / (index + 1)

def update_loss_MA(loss_MA, all_losses):
    loss_MA_alpha = 0.60
    if loss_MA is None:
        loss_MA = all_losses[-1]
    else:
        loss_MA = loss_MA_alpha * loss_MA + (1 - loss_MA_alpha) * all_losses[-1]
    return loss_MA

def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create network.
        # deeplab = UNet(input_size, num_classes=args.num_classes, weight_std=args.weight_std)
        model = EfficientSegBackbone(num_classes=args.num_classes, compound_coef=args.compound_coef, load_weights=True)
        print_model_parm_nums(model)
        # print_model_parm_flops(deeplab)
        # torch_summarize_df(input_size=(1, 16, 224, 224), model=deeplab)

        # saved_state_dict = torch.load(args.restore_from)
        # deeplab.load_state_dict(saved_state_dict)

        model.train()
        if args.num_gpus > 1:
            model = convert_syncbn_model(model)

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, nesterov=True)
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                               patience=args.patience, verbose=True, threshold=1e-3,
                                                               threshold_mode='abs')

        if args.FP16:
            print("Note: Using FP16 during training************")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        loss_seg_DICE = loss.DiceLoss().to(device)
        loss_seg_CE = loss.CELoss().to(device)
        loss_seg_surface = loss.SurfaceLoss(idc=[4, 5]).to(device)
        # loss_seg_DICE = DataParallelCriterion(loss_seg_DICE)
        # loss_seg_surface = DataParallelCriterion(loss_seg_surface)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        trainloader, train_sampler = engine.get_train_loader(
            MyoPSDataSet(args.data_dir, args.train_list, max_iters=100 * args.batch_size,
                         crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror))

        valloader, val_sampler = engine.get_test_loader(
            MyoPSDataSetVal(args.data_dir, args.val_list))

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999

        for epoch in range(args.num_epochs):
            if epoch < args.start_epoch:
                continue

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []

            for i_iter, batch in enumerate(trainloader):

                images, labels, _, volumeName, distance_map = batch
                images = images.cuda()
                labels = labels.cuda()
                distance_map = distance_map.cuda()

                optimizer.zero_grad()
                preds = model(images)

                term_seg_Dice = loss_seg_DICE.forward(preds, labels)
                term_seg_CE = loss_seg_CE.forward(preds, labels)
                term_seg_surface = loss_seg_surface.forward(preds, distance_map)

                alpha = adjust_alpha(epoch, args.num_epochs)
                term_all = alpha*(term_seg_Dice+term_seg_CE) + (1-alpha)*term_seg_surface

                reduce_seg_Dice = engine.all_reduce_tensor(term_seg_Dice)
                reduce_seg_CE = engine.all_reduce_tensor(term_seg_CE)
                reduce_seg_surface = engine.all_reduce_tensor(term_seg_surface)
                reduce_all = engine.all_reduce_tensor(term_all)

                if args.FP16:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    term_all.backward()
                optimizer.step()

                epoch_loss.append(float(reduce_all))

                if (args.local_rank == 0):
                    print('Epoch {} = {}/{}, lr = {:.4}, loss = {:.4}, loss_Dice = {:.4}, loss_CE = {:.4}, loss_Surface = {:.4}'.format(
                            epoch, i_iter, len(trainloader), optimizer.param_groups[0]['lr'],
                            reduce_all.cpu().data.numpy(),
                            reduce_seg_Dice.cpu().data.numpy(),
                            reduce_seg_CE.cpu().data.numpy(),
                            reduce_seg_surface.cpu().data.numpy()))

            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss.item()))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', epoch_loss.item(), epoch)

            # val
            if (epoch > 0 and (epoch % args.save_pred_every == 0)) or (epoch >= args.num_epochs -1):
                print('validate ...')
                val_loss, val_LV, val_RV, val_MY, val_edema, val_scars = validate(args, input_size, model, valloader, args.num_classes, loss_seg_DICE, loss_seg_surface, engine)

                # save model according to val_loss
                if val_best_loss > val_loss:
                    val_best_loss = val_loss
                    if args.local_rank == 0:
                        print('Finding best model in epoch {}, saving...'.format(epoch))
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'MyoPS2020_' + args.snapshot_dir.split('/')[-2] + '_best.pth'))
                all_va_loss.append(val_loss)
                val_loss_MA = update_loss_MA(val_loss_MA, all_va_loss)
                scheduler.step(val_loss_MA)
                if (args.local_rank == 0):
                    writer.add_scalar('Val_loss', val_loss.item(), epoch)
                    writer.add_scalar('Val_MA_loss', val_loss_MA.item(), epoch)
                    writer.add_scalar('Val_LV_Dice', val_LV, epoch)
                    writer.add_scalar('Val_RV_Dice', val_RV, epoch)
                    writer.add_scalar('Val_MY_Dice', val_MY, epoch)
                    writer.add_scalar('Val_edema_Dice', val_edema, epoch)
                    writer.add_scalar('Val_scars_Dice', val_scars, epoch)
                    print('Validate epoch = {}, loss={:.4}, LV = {:.4}, RV = {:.4}, MY = {:.4}, edema = {:.4}, scars = {:.4}'.format(epoch,
                                                                                                                                    val_loss.item(),
                                                                                                                                    val_LV.item(),
                                                                                                                                    val_RV.item(),
                                                                                                                                    val_MY.item(),
                                                                                                                                    val_edema.item(),
                                                                                                                                    val_scars.item()))
            # if epoch >= args.num_epochs - 1 and (args.local_rank == 0):
            #     print('save model ...')
            #     torch.save(model.state_dict(), osp.join(args.snapshot_dir,
            #                                             'MyoPS2020_' + args.snapshot_dir.split('/')[-2] + '_' + str(
            #                                                 args.num_epochs) + '.pth'))
            #     break
            if optimizer.param_groups[0]['lr'] < 1e-6 and (args.local_rank == 0):
                print("Reprot: lr is less than 1e-6! Breaking...")
                print('save model ...')
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'MyoPS2020_' + args.snapshot_dir.split('/')[-2] + '_final.pth'))
                break

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
