# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import networks.deeplab.resnet_atrous as atrousnet
import networks.deeplab.xception as xception

def build_backbone(backbone_name, pretrained=True, os=16):
	if backbone_name == 'res50_atrous':
		model = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
		return model
	elif backbone_name == 'res101_atrous':
		model = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
		return model
	elif backbone_name == 'res152_atrous':
		model = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
		return model
	elif backbone_name == 'xception' or backbone_name == 'Xception':
		model = xception.xception(pretrained=pretrained, os=os)
		return model
	elif backbone_name == 'xception_paral' or backbone_name == 'Xception_paral':
		model = xception_paral.xception(pretrained=pretrained, os=os)
		return model
	else:
		raise ValueError('backbone.py: The backbone named %s is not supported yet.'%backbone_name)
	

	

