import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import label as LAB
from skimage.transform import resize
import SimpleITK as sitk



img_folder_path = 'output/MyoPS2020/test'

# edema-1220       scars-2221
def continues_region_extract(label, class_):
    numbers = []
    label_pp = label.copy()
    for i in range(label_pp.shape[2]):
        label_i = label_pp[:,:,i]
        regions = np.where(label_i==class_, np.ones_like(label_i), np.zeros_like(label_i))
        L_i, n_i = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

        for j in range(1, n_i + 1):
            num_j = np.sum(L_i == j)
            numbers.append(num_j)
            if num_j<60:
                bbx_h, bbx_w = np.where(L_i==j)
                bbx_h_min = bbx_h.min()
                bbx_h_max = bbx_h.max()
                bbx_w_min = bbx_w.min()
                bbx_w_max = bbx_w.max()
                roi = label_i[bbx_h_min-1:bbx_h_max+2, bbx_w_min-1:bbx_w_max+2]
                replace_lable = np.argmax(np.bincount(roi[roi!=class_].flatten()))

                label_pp[:,:,i] = np.where(L_i==j, replace_lable*np.ones_like(label_i), label_i)

    return numbers, label_pp


def continues_region_extract2(label):
    numbers = []
    label_pp = label.copy()
    for i in range(label_pp.shape[2]):
        label_i = label_pp[:,:,i]
        regions = np.where(label_i>=1220, np.ones_like(label_i), np.zeros_like(label_i))
        L_i, n_i = LAB(regions, neighbors=4, background=0, connectivity=2, return_num=True)

        for j in range(1, n_i + 1):
            num_j = np.sum(L_i == j)
            numbers.append(num_j)
            if num_j<200:
                bbx_h, bbx_w = np.where(L_i==j)
                bbx_h_min = bbx_h.min()
                bbx_h_max = bbx_h.max()
                bbx_w_min = bbx_w.min()
                bbx_w_max = bbx_w.max()
                roi = label_i[bbx_h_min-1:bbx_h_max+2, bbx_w_min-1:bbx_w_max+2]
                replace_lable = np.argmax(np.bincount(roi[roi<1220].flatten()))

                label_pp[:,:,i] = np.where(L_i==j, replace_lable*np.ones_like(label_i), label_i)

    return numbers, label_pp

for root, dirs, files in os.walk(img_folder_path):
    for i in sorted(files):
        if i[-2:] != 'gz' or i[0]=='P':
            continue
        i_file = root +'/'+ i
        predNII = nib.load(i_file)
        label = predNII.get_data()

        numbers1, label_pp = continues_region_extract(label, 2221)
        # numbers2, label_pp = continues_region_extract(label_pp, 1220)
        numbers3, label_pp = continues_region_extract2(label_pp)
        # print("%s: %s" % (i[15:18], numbers))
        # print("%s: %s" % (i[11:14], numbers))

        label_pp = label_pp.astype(np.int16)

        label_pp = nib.Nifti1Image(label_pp, affine=predNII.affine)
        name = img_folder_path + '/PP_' + i
        seg_save_p = os.path.join(name)
        nib.save(label_pp, seg_save_p)

        pass
