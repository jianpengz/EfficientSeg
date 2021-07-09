# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from networks.deeplab.backbone import build_backbone


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch3 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch4 = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out)
		self.branch5_relu = nn.ReLU(inplace=True)
		self.conv_cat = nn.Sequential(
			nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
			nn.BatchNorm2d(dim_out),
			nn.ReLU(inplace=True),
		)


	def forward(self, x):
		[b, c, row, col] = x.size()
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
		global_feature = torch.mean(x, 2, True)
		global_feature = torch.mean(global_feature, 3, True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result


class deeplabv3plus(nn.Module):
	def __init__(self, num_classes=None, OS=16):
		super(deeplabv3plus, self).__init__()
		self.MODEL_NUM_CLASSES = num_classes
		self.backbone = build_backbone('Xception', os=OS)
		self.backbone_layers = self.backbone.get_layers()
		self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//OS, bn_mom = 0.99)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=OS//4)

		self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)
		self.cat_conv = nn.Sequential(
				nn.Conv2d(256+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, self.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

class PRM(nn.Module):
	def __init__(self):
		super(PRM, self).__init__()
		self.softmax = nn.Softmax(dim=-1)
		self.MSE_loss = nn.MSELoss()
		self.Smooth_L1Loss = nn.SmoothL1Loss()

	def forward(self, l_p, ul_p, num_l):

		_, C, height, width = ul_p.size()

		proj_query = ul_p.view(num_l, -1, width * height).permute(0, 2, 1)
		proj_key = l_p.view(num_l, -1, width * height)

		energy = torch.bmm(proj_query / proj_key.max(), proj_key / proj_key.max())
		energy = energy.to(torch.float32) * proj_key.max() * proj_key.max()

		# energy = torch.bmm(proj_query, proj_key)

		attention = self.softmax(energy)
		proj_value = l_p.view(num_l, -1, width * height)

		attentionM = torch.bmm(proj_value, attention.permute(0, 2, 1))
		attentionM = attentionM.view(ul_p.size()[0], C, height, width)

		W_ul = attentionM + ul_p

		return W_ul


class deeplabv3plus_PRM(nn.Module):
	def __init__(self, num_classes=None):
		super(deeplabv3plus_PRM, self).__init__()
		self.MODEL_NUM_CLASSES = num_classes
		self.backbone = None
		self.backbone_layers = None
		self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)

		self.PRM_module = PRM()

		self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)
		self.cat_conv = nn.Sequential(
				nn.Conv2d(256+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, self.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.backbone = build_backbone('Xception', os=16)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x, num_l):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])

		p_x1 = feature_aspp[0:num_l, :, :, :]
		p_x2 = feature_aspp[num_l:num_l * 2, :, :, :]

		feature_aspp_argu_2 = self.PRM_module(p_x1, p_x2, num_l)
		feature_aspp_argu_1 = self.PRM_module(p_x2, p_x1, num_l)

		feature_aspp_argu = torch.cat([feature_aspp_argu_1,feature_aspp_argu_2], dim=0)

		feature_aspp = self.dropout1(feature_aspp_argu)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp, feature_shallow],1)
		result = self.cat_conv(feature_cat)
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

class deeplabv3plus_2(nn.Module):
	def __init__(self, num_classes=None):
		super(deeplabv3plus_2, self).__init__()
		self.MODEL_NUM_CLASSES = num_classes
		self.backbone = None
		self.backbone_layers = None
		self.aspp = ASPP(dim_in=2048, dim_out=256, rate=16//16, bn_mom = 0.99)
		self.dropout1 = nn.Dropout(0.5)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)

		self.shortcut_conv = nn.Sequential(nn.Conv2d(256, 48, 1, 1, padding=1//2, bias=True),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),
		)
		self.cat_conv = nn.Sequential(
				nn.Conv2d(256+48, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1, bias=True),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, self.MODEL_NUM_CLASSES, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.backbone = build_backbone('Xception', os=16)
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp_out = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp_out)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat)
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result, feature_aspp_out


