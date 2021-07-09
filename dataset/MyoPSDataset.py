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
from skimage.transform import resize
import SimpleITK as sitk
import time
from loss_functions.surface import one_hot2dist


class MyoPSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            filepath, gtpath, depth_sample = item
            C0_path = filepath + '_C0.nii.gz'
            DE_path = filepath + '_DE.nii.gz'
            T2_path = filepath + '_T2.nii.gz'
            label_path = gtpath + '_gd.nii.gz'
            # image_path, label_path = item
            name = osp.splitext(osp.basename(filepath))[0]
            # img_file = osp.join(self.root, image_path)
            C0_file = osp.join(self.root, C0_path)
            DE_file = osp.join(self.root, DE_path)
            T2_file = osp.join(self.root, T2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "C0": C0_file,
                "DE": DE_file,
                "T2": T2_file,
                "label": label_file,
                "name": name,
                "depth_sample": int(depth_sample)
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        # left ventricular (LV) blood pool (labelled 500),
        # right ventricular blood pool (600),
        # LV normal myocardium (200),
        # LV myocardial edema (1220),
        # LV myocardial scars (2221),
        shape = label.shape
        results_map = np.zeros((5+1, shape[0], shape[1]))

        LV = (label == 500)
        RV = (label == 600)
        MY = (label == 200)
        edema = (label == 1220)
        scars = (label == 2221)

        background = np.logical_not(LV + RV + MY + edema + scars)

        results_map[0, :, :] = np.where(background, 1, 0)
        results_map[1, :, :] = np.where(LV, 1, 0)
        results_map[2, :, :] = np.where(RV, 1, 0)
        results_map[3, :, :] = np.where(MY, 1, 0)
        results_map[4, :, :] = np.where(edema, 1, 0)
        results_map[5, :, :] = np.where(scars, 1, 0)
        return results_map

    def truncate(self, MRI):
        # truncate
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))

        idexs = np.argwhere(Hist >= 20)
        idex_min = np.float32(0)
        idex_max = np.float32(idexs[-1, 0])

        # MRI[np.where(MRI <= idex_min)] = idex_min
        MRI[np.where(MRI >= idex_max)] = idex_max
        # MRI = MRI - (idex_max+idex_min)/2
        # MRI = MRI / ((idex_max-idex_min)/2)

        # norm
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        C0NII = nib.load(datafiles["C0"])
        DENII = nib.load(datafiles["DE"])
        T2NII = nib.load(datafiles["T2"])
        labelNII = nib.load(datafiles["label"])

        C0 = self.truncate(C0NII.get_data())
        DE = self.truncate(DENII.get_data())
        T2 = self.truncate(T2NII.get_data())

        image = np.array([C0, DE, T2])  # 3x
        label = labelNII.get_data()
        size = image.shape
        name = datafiles["name"]
        depth_sample = datafiles["depth_sample"]

        # nib.save(nib.Nifti1Image(image[0].astype(np.int16), affine=flairNII.affine), "aab_flair.nii.gz")

        # pre-processing
        image = image[:,:,:,depth_sample].astype(np.float32)
        label = label[:,:,depth_sample].astype(np.float32)

        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)  # 0.9, 1.1
        else:
            scaler = 1
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w = label.shape
        #print("%s, scaleh:%d, scale_w:%d, imgh:%d, imgw:%d, img_d:%d" % (name, scale_h, scale_w, img_h, img_w, img_d))
        h_off = random.randint(0, img_h - 0 - scale_h)
        w_off = random.randint(0, img_w - 0 - scale_w)
        # h_off = 10
        # w_off = 10

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w]

        # get result map
        label = self.id2trainId(label)

        # image = image.transpose((1, 2, 0))  # Channel x H x W
        # label = label.transpose((0, 3, 1, 2))  # Depth x H x W

        if self.is_mirror:
            randi = np.random.rand(1)
            # 0:0.4, 1:0.3, 2:0.3
            if randi <= 0.4:
                pass
            elif randi <= 0.7:  # flip W
                image = image[:, :, ::-1]
                label = label[:, :, ::-1]
            else:
                image = image[:, ::-1, :]
                label = label[:, ::-1, :]

        if self.scale:
            image = resize(image, (3, self.crop_h, self.crop_w), order=1, mode='constant', cval=0,
                           clip=True, preserve_range=True)
            label = resize(label, (5+1, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True,
                           preserve_range=True)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        distance_map = one_hot2dist(label)

        return image.copy(), label.copy(), np.array(size), name, distance_map


class MyoPSDataSetVal(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            filepath, gtpath = item
            C0_path = filepath + '_C0.nii.gz'
            DE_path = filepath + '_DE.nii.gz'
            T2_path = filepath + '_T2.nii.gz'
            label_path = gtpath + '_gd.nii.gz'
            # image_path, label_path = item
            name = osp.splitext(osp.basename(filepath))[0]
            # img_file = osp.join(self.root, image_path)
            C0_file = osp.join(self.root, C0_path)
            DE_file = osp.join(self.root, DE_path)
            T2_file = osp.join(self.root, T2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "C0": C0_file,
                "DE": DE_file,
                "T2": T2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        # left ventricular (LV) blood pool (labelled 500),
        # right ventricular blood pool (600),
        # LV normal myocardium (200),
        # LV myocardial edema (1220),
        # LV myocardial scars (2221),
        shape = label.shape
        results_map = np.zeros((5+1, shape[0], shape[1], shape[2]))

        LV = (label == 500)
        RV = (label == 600)
        MY = (label == 200)
        edema = (label == 1220)
        scars = (label == 2221)

        background = np.logical_not(LV + RV + MY + edema + scars)

        results_map[0, :, :, :] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(LV, 1, 0)
        results_map[2, :, :, :] = np.where(RV, 1, 0)
        results_map[3, :, :, :] = np.where(MY, 1, 0)
        results_map[4, :, :, :] = np.where(edema, 1, 0)
        results_map[5, :, :, :] = np.where(scars, 1, 0)
        return results_map

    def truncate(self, MRI):
        # truncate
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))

        idexs = np.argwhere(Hist >= 20)
        idex_min = np.float32(0)
        idex_max = np.float32(idexs[-1, 0])

        # MRI[np.where(MRI <= idex_min)] = idex_min
        MRI[np.where(MRI >= idex_max)] = idex_max
        # MRI = MRI - (idex_max+idex_min)/2
        # MRI = MRI / ((idex_max-idex_min)/2)

        # norm
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        C0NII = nib.load(datafiles["C0"])
        DENII = nib.load(datafiles["DE"])
        T2NII = nib.load(datafiles["T2"])
        labelNII = nib.load(datafiles["label"])

        C0 = self.truncate(C0NII.get_data())
        DE = self.truncate(DENII.get_data())
        T2 = self.truncate(T2NII.get_data())

        image = np.array([C0, DE, T2])  # 3x
        label = labelNII.get_data()
        size = image.shape
        name = datafiles["name"]

        # nib.save(nib.Nifti1Image(image[0].astype(np.int16), affine=flairNII.affine), "aab_flair.nii.gz")

        # get result map
        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))  # Depth x H x W

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), np.array(size), name, np.array(labelNII.affine)


if __name__ == '__main__':
    dst = MyoPSDataSet("./", 'list/MyoPS2020/train5f_1.txt', max_iters=40000*2)
    trainloader = data.DataLoader(dst, shuffle=False, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, sizes, names = data
        print("%s; slices:%d" % (names, imgs.shape[2]))
        if False:  # i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()  # 4,155, 240, 240
            # img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, ::-1]
            plt.imshow(img[0, 80, 0:160, 0:192])
            plt.show()

            t = labels.shape
            g_t = torchvision.utils.make_grid(labels).numpy()
            # g_t = np.transpose(g_t, (1, 2, 0))
            plt.imshow(g_t[1, 80, 0:160, 0:192])
            plt.show()