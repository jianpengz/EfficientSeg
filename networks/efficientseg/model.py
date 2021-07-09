import torch.nn as nn
import torch
import torch.nn.functional as F

from networks.efficientnet import EfficientNet as EffNet
from networks.efficientnet.utils import MemoryEfficientSwish, Swish
from networks.efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv2_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv1_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv2_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[4], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p2_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p1_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # self.p3_to_p4 = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            #     MaxPool2dStaticSamePadding(3, 2)
            # )
            # self.p4_to_p5 = nn.Sequential(
            #     MaxPool2dStaticSamePadding(3, 2)
            # )

            self.p2_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()

        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P5_0 -------------------------> P5_2 -------->
               |-------------|                ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P3_0 ---------> P3_1 ---------> P3_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P2_0 ---------> P2_1 ---------> P2_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P1_0 -------------------------> P1_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            p1_out, p2_out, p3_out, p4_out, p5_out = self._forward_fast_attention(inputs)
        else:
            p1_out, p2_out, p3_out, p4_out, p5_out = self._forward(inputs)

        return p1_out, p2_out, p3_out, p4_out, p5_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p1, p2, p3, p4, p5 = inputs

            # p4_in = self.p3_to_p4(p3)
            # p5_in = self.p4_to_p5(p4_in)

            p1_in = self.p1_down_channel(p1)
            p2_in = self.p2_down_channel(p2)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P1_0, P2_0, P3_0, P4_0 and P5_0
            p1_in, p2_in, p3_in, p4_in, p5_in = inputs

        # P5_0 to P5_2

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_in)))

        # Weights for P3_0 and P4_0 to P3_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_0 to P3_1 respectively
        p3_up = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        # Weights for P2_0 and P3_0 to P2_1
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for P2_0 and P3_0 to P2_1 respectively
        p2_up = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p2_upsample(p3_up)))

        # Weights for P1_0 and P2_1 to P1_2
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        # Connections for P1_0 and P2_1 to P1_2 respectively
        p1_out = self.conv1_up(self.swish(weight[0] * p1_in + weight[1] * self.p1_upsample(p2_up)))

        # if self.first_time:
        #     p2_in = self.p2_down_channel_2(p2)
        #     p3_in = self.p3_down_channel_2(p3)

        # Weights for P2_0, P2_1 and P1_2 to P2_2
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        # Connections for P2_0, P2_1 and P1_2 to P2_2 respectively
        p2_out = self.conv2_down(
            self.swish(weight[0] * p2_in + weight[1] * p2_up + weight[2] * self.p2_downsample(p1_out)))

        # Weights for P3_0, P3_1 and P2_2 to P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
        p3_out = self.conv3_down(
            self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p3_downsample(p2_out)))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0 and P4_2 to P5_2
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * self.p5_downsample(p4_out)))

        return p1_out, p2_out, p3_out, p4_out, p5_out

    def _forward(self, inputs):
        if self.first_time:
            p1, p2, p3, p4, p5 = inputs

            # p4_in = self.p3_to_p4(p3)
            # p5_in = self.p4_to_p5(p4_in)

            p1_in = self.p1_down_channel(p1)
            p2_in = self.p2_down_channel(p2)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            p1_in, p2_in, p3_in, p4_in, p5_in = inputs

        # P5_0 to P5_2

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_in)))

        # Connections for P3_0 and P4_0 to P3_1 respectively
        p3_up = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        # Connections for P2_0 and P3_0 to P2_1 respectively
        p2_up = self.conv2_up(self.swish(p2_in + self.p2_upsample(p3_up)))

        # Connections for P1_0 and P2_1 to P1_2 respectively
        p1_out = self.conv1_up(self.swish(p1_in + self.p1_upsample(p2_up)))

        if self.first_time:
            p2_in = self.p2_down_channel_2(p2)
            p3_in = self.p3_down_channel_2(p3)

        # Connections for P2_0, P2_1 and P1_2 to P2_2 respectively
        p2_out = self.conv2_down(
            self.swish(p2_in + p2_up + self.p2_downsample(p1_out)))

        # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
        p3_out = self.conv3_down(
            self.swish(p3_in + p3_up + self.p3_downsample(p2_out)))

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0 and P4_2 to P5_2
        p5_out = self.conv5_down(self.swish(p5_in + self.p5_downsample(p4_out)))

        return p1_out, p2_out, p3_out, p4_out, p5_out


class Segmentor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_classes, num_layers, onnx_export=False):
        super(Segmentor, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)])
        self.header = SeparableConvBlock(in_channels, num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        p1,p2,p3,p4,p5 = inputs
        p2 = F.upsample(p2, size=(p1.size(2), p1.size(3)), mode='bilinear')
        p3 = F.upsample(p3, size=(p1.size(2), p1.size(3)), mode='bilinear')
        p4 = F.upsample(p4, size=(p1.size(2), p1.size(3)), mode='bilinear')
        p5 = F.upsample(p5, size=(p1.size(2), p1.size(3)), mode='bilinear')

        # feat = torch.cat([p1,p2,p3,p4,p5],1)
        feat = p1+p2+p3+p4+p5

        for i, bn, conv in zip(range(self.num_layers), self.bn_list, self.conv_list):
            feat = conv(feat)
            feat = bn(feat)
            feat = self.swish(feat)

        feat = self.header(feat)
        output = F.upsample(feat, size=(p1.size(2)*2, p1.size(3)*2), mode='bilinear')

        return output

class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps


if __name__ == '__main__':
    from tensorboardX import SummaryWriter


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
