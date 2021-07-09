import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
# from networks.unet2D import UNet
from networks.efficient import EfficientSegBackbone
from dataset.MyoPSDataset import MyoPSDataSet, MyoPSDataSetVal
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
import nibabel as nib
from utils.ParaFlop import print_model_parm_nums, print_model_parm_flops, torch_summarize_df
import matplotlib.pyplot as plt
# from utils.encoding import DataParallelModel, DataParallelCriterion
import torch.nn as nn
from engine import Engine


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="W-like Network for 3D Medical Image Segmentation.")

    # paths
    parser.add_argument("--data_dir", type=str, default='dataset/')
    parser.add_argument("--val_list", type=str, default='list/MyoPS2020/val5f_5.txt')
    parser.add_argument("--output_path", type=str, default='output/MyoPS2020/tmp/')
    parser.add_argument("--restore_from", type=str, default='snapshots/MyoPS2020/all_r1_EB3_4GPU_bs32_FP16/MyoPS2020_all_r1_EB3_4GPU_bs32_FP16_final.pth')

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
    parser.add_argument("--num_classes", type=int, default=5 + 1)
    parser.add_argument("--compound_coef", type=int, default=3)  # B0-B7
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

    return parser

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(img, ((0, 0), (0, 0),(0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def multi_net(net_list, img):
    img = torch.from_numpy(img).cuda()
    padded_prediction = net_list[0](img)
    padded_prediction = torch.softmax(padded_prediction, 1)

    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img)
        padded_prediction_i = torch.softmax(padded_prediction_i, 1)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction.cpu().numpy()

# for 2D
def predict_sliding(net, image, tile_size, classes):  # image: 1,3,5,256,256, tile_size:256x256
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((image_size[3] - tile_size[0]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[1]) / strideHW) + 1)
    # print("Need %i x %i x %i prediction tiles @ stride %i x %i px" % (tile_deps, tile_cols, tile_rows, strideD, strideHW))
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
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
                img = img.numpy()

                tile_counter += 1
                prediction1 = multi_net([net], img)
                prediction2 = multi_net([net], img[:, :, :, ::-1].copy())[:, :, :, ::-1]
                prediction3 = multi_net([net], img[:, :, ::-1, :].copy())[:, :, ::-1, :]
                prediction = (prediction1 + prediction2 + prediction3) / 3.

                prediction = torch.unsqueeze(torch.from_numpy(prediction).cuda(), 2)

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
    return full_probs.cpu().data.numpy()



def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix

def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) +1

    dice = 2*num / den

    return dice.mean()

def main():
    """Create the model and start the evaluation process."""
    parser= get_arguments()
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        torch.cuda.set_device(args.local_rank)

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = EfficientSegBackbone(num_classes=args.num_classes, compound_coef=args.compound_coef, load_weights=True)
        print_model_parm_nums(model)

        # print_model_parm_flops(model)
        # model = nn.DataParallel(model)

        print('loading from checkpoint: {}'.format(args.restore_from))
        if os.path.exists(args.restore_from):
            model.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
            #model.load_state_dict(torch.load(args.restore_from, map_location=torch.device(args.local_rank)))
        else:
            print('File not exists in the reload path: {}'.format(args.restore_from))

        model.eval()
        model.cuda()

        testloader = data.DataLoader(
            MyoPSDataSetVal(args.data_dir, args.val_list),
            batch_size=1, shuffle=False, pin_memory=True)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        dice_LV = 0
        dice_RV = 0
        dice_MY = 0
        dice_edema = 0
        dice_scars = 0

        for index, batch in enumerate(testloader):
            # print('%d processd'%(index))
            image, label, name, affine = batch
            affine = affine[0].numpy()
            with torch.no_grad():
                output = predict_sliding(model, image, input_size, args.num_classes)

            seg_pred = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
            seg_pred = np.where(seg_pred == 1, 500, seg_pred)
            seg_pred = np.where(seg_pred == 2, 600, seg_pred)
            seg_pred = np.where(seg_pred == 3, 200, seg_pred)
            seg_pred = np.where(seg_pred == 4, 1220, seg_pred)
            seg_pred = np.where(seg_pred == 5, 2221, seg_pred)

            seg_pred_background = (seg_pred == 0)
            seg_pred_LV = (seg_pred == 500)
            seg_pred_RV = (seg_pred == 600)
            seg_pred_MY = (seg_pred == 200)
            seg_pred_edema = (seg_pred == 1220)
            seg_pred_scars = (seg_pred == 2221)

            seg_gt = np.asarray(np.argmax(label.cpu().numpy(), axis=1), dtype=np.uint8)
            seg_gt = np.where(seg_gt == 1, 500, seg_gt)
            seg_gt = np.where(seg_gt == 2, 600, seg_gt)
            seg_gt = np.where(seg_gt == 3, 200, seg_gt)
            seg_gt = np.where(seg_gt == 4, 1220, seg_gt)
            seg_gt = np.where(seg_gt == 5, 2221, seg_gt)

            seg_gt_background = (seg_gt == 0)
            seg_gt_LV = (seg_gt == 500)
            seg_gt_RV = (seg_gt == 600)
            seg_gt_MY = (seg_gt == 200)
            seg_gt_edema = (seg_gt == 1220)
            seg_gt_scars = (seg_gt == 2221)

            dice_LV_i = dice_score(seg_pred_LV, seg_gt_LV)
            dice_RV_i = dice_score(seg_pred_RV, seg_gt_RV)
            dice_MY_i = dice_score(seg_pred_MY, seg_gt_MY)
            dice_edema_i = dice_score(seg_pred_edema, seg_gt_edema)
            dice_scars_i = dice_score(seg_pred_scars, seg_gt_scars)
            print('Processing {}: LV = {:.4}, RV = {:.4}, MY = {:.4}, edema = {:.4}, scars = {:.4}'.format(name, dice_LV_i, dice_RV_i, dice_MY_i, dice_edema_i, dice_scars_i))

            dice_LV += dice_LV_i
            dice_RV += dice_RV_i
            dice_MY += dice_MY_i
            dice_edema += dice_edema_i
            dice_scars += dice_scars_i

            seg_pred = seg_pred[0].transpose((1,2,0))   #240x240x155
            seg_gt = seg_gt[0].transpose((1,2,0))

            seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
            seg_gt = nib.Nifti1Image(seg_gt, affine=affine)
            # seg_name = name[0].replace("volume", "segmentation")
            seg_name = name[0]
            seg_save_p = os.path.join(args.output_path+'/%s.nii.gz' % (seg_name))
            gt_save_p = os.path.join(args.output_path + '/%s_gt.nii.gz' % (seg_name))
            nib.save(seg_pred, seg_save_p)
            nib.save(seg_gt, gt_save_p)

        dice_LV_avg = dice_LV / (index + 1)
        dice_RV_avg = dice_RV / (index + 1)
        dice_MY_avg = dice_MY / (index + 1)
        dice_edema_avg = dice_edema / (index + 1)
        dice_scanrs_avg = dice_scars / (index + 1)
        print('Average score: LV = {:.4}, RV = {:.4}, MY = {:.4}, edema = {:.4}, scars = {:.4}'.format(dice_LV_avg, dice_RV_avg, dice_MY_avg, dice_edema_avg, dice_scanrs_avg))


if __name__ == '__main__':
    main()