import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum
from loss_functions.surface import simplex, one_hot
from typing import List

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


class GeneralizedDice(nn.Module):
    def __init__(self):
        super(GeneralizedDice, self).__init__()

    def forward(self, predict, target):
        # target = target[:, None, :, :, :]
        # target = make_one_hot(target, predict.shape[1])
        predict = torch.softmax(predict,1)

        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        w = 1 / ((einsum("bcwh->bc", target).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", predict, target)
        union= w * (einsum("bcwh->bc", predict) + einsum("bcwh->bc", target))

        divided = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss

class SurfaceLoss(nn.Module):
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        super(SurfaceLoss, self).__init__()
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, probs, dist_maps):
        probs = torch.softmax(probs, 1)
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # sig = torch.sum(target, dim=1)
        # n = len(sig.nonzero())

        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        loss_avg = 1 - dice_score

        return loss_avg


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        bs, num_cls, H, W = predict.shape
        assert predict.shape == target.shape, 'predict & target shape do not match' # (1,5,3,256,256)
        dice = BinaryDiceLoss(**self.kwargs)
        predict = torch.softmax(predict, dim=1)

        predict = predict[:,1:]
        target = target[:,1:]
        predict = predict.contiguous().view(-1, H, W)
        target = target.contiguous().view(-1, H, W)
        dice_loss = dice(predict, target)

        total_loss = dice_loss.mean()

        return total_loss

class CELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CELoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def weight_function(self, mask):
        weights = torch.ones_like(mask).float()
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        for i in range(mask.max()+1):
            voxels_i = [mask==i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum/voxels_i).astype(np.float32)
            weights = torch.where(mask==i, w_i*torch.ones_like(weights).float(), weights)

        return weights

    def forward(self, predict, target):
        # target = target.cpu()
        # predict = predict.cpu()
        #onehot
        # target = target[:,None,:,:,:]
        # target = make_one_hot(target, predict.shape[1])
        # bs, num_cls, depth, H, W = predict.shape
        assert predict.shape == target.shape, 'predict & target shape do not match' # (1,6,3,256,256)
        target = torch.argmax(target, 1)
        weights = self.weight_function(target)

        ce_loss = self.criterion(predict, target)

        weight_ce_loss = ce_loss * weights

        return weight_ce_loss.mean()
