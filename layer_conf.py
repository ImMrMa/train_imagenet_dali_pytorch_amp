import torch.nn as nn
from torch.nn.parameter import Parameter
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
import torch
import os.path as osp
import math
__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]

model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def group_norm(in_planes):
    return nn.GroupNorm(in_planes // 16, in_planes)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv4x4(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=4,
                     stride=stride,
                     padding=padding,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=0):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=2,
                     stride=stride,
                     padding=padding,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class L2Norm(nn.Module):
    def __init__(self, inplanes, gamma_init=10):
        super(L2Norm, self).__init__()
        self.gamma_init = torch.Tensor(1, inplanes, 1, 1)
        self.gamma_init[...] = gamma_init
        self.gamma_init = Parameter(self.gamma_init)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = x * self.gamma_init
        return x


class InceptionBlock(nn.Module):
    def __init__(self, multi_large=False,):
        super(InceptionBlock, self).__init__()
        self.multi_large = multi_large

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False),
             nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,
                       stride=1, padding=1, bias=False),
             nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5,
                       stride=1, padding=2, bias=False),
             nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)]
        )

    def forward(self, x):
        output = []
        for conv_module in self.convs:
            output.append(conv_module(x))
        x = torch.cat(output, 1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 replace_with_bn=False,
                 layer_conv=None):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if layer_conv:
            self.conv1 = nn.Conv2d(inplanes,
                                   planes,
                                   kernel_size=layer_conv['k'],
                                   stride=layer_conv['s'],
                                   padding=layer_conv['p'],
                                   groups=groups,
                                   bias=False,
                                   dilation=layer_conv['d'])
        else:
            self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if replace_with_bn:
            norm_layer = group_norm
        self.bn1 = norm_layer(planes)
        norm_layer = self.norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 replace_with_bn=False,
                 layer_conv=None):
        super(Bottleneck, self).__init__()
        self.norm_layer = norm_layer
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if layer_conv:
            self.conv2 = nn.Conv2d(width,
                                   width,
                                   kernel_size=layer_conv['k'],
                                   stride=layer_conv['s'],
                                   padding=layer_conv['p'],
                                   groups=groups,
                                   bias=False,
                                   dilation=layer_conv['d'])
            stride=layer_conv['s']
        else:
            self.conv2 = conv3x3(width, width, stride, dilation=dilation)
        if replace_with_bn:
            norm_layer = group_norm
        self.bn2 = norm_layer(width)
        norm_layer = self.norm_layer
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 layer_conf=dict(),
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 change_s1=False,
                 replace_with_bn=False,
                 all_gn=False, mulit_stage=False,classification=True):
        super(ResNet, self).__init__()
        self.classification=classification
        self.mulit_stage = mulit_stage
        self.layer_conf = dict(
            layer0=dict(conv=dict(k=4, p=1, d=1, s=2),
                        down=dict(k=2, p=0, d=1, s=2)),
            layer1=dict(conv=dict(k=4, p=1, d=1, s=2),
                        down=dict(k=2, p=0, d=1, s=2)),
            layer2=dict(conv=dict(k=4, p=1, d=1, s=2),
                        down=dict(k=2, p=0, d=1, s=2)),
            layer3=dict(conv=dict(k=4, p=1, d=1, s=2),
                        down=dict(k=2, p=0, d=1, s=2)),
            layer4=dict(conv=dict(k=4, p=3, d=2, s=1),
                        down=dict(k=1, p=0, d=1, s=1)))
        layer_conf_conv4_conv2 = dict(conv=dict(k=4, p=1, d=1, s=2),
                                      down=dict(k=2, p=0, d=1, s=2))
        self.layer_conf.update(layer_conf)
        layer_conf = self.layer_conf
        if all_gn:
            bn_layer = group_norm
        else:
            bn_layer = nn.BatchNorm2d

        norm_layer = bn_layer
        self._norm_layer = norm_layer

        if block is BasicBlock:
            transform_planes = 128
        elif block is Bottleneck:
            transform_planes = 256
        else:
            raise ValueError('not block')
        self.inplanes = 64
        self.dilation = 1
        self.change_s1 = change_s1
        self.groups = groups
        self.base_width = width_per_group
        if change_s1:
            if replace_with_bn:
                norm_layer = group_norm
            self.inplanes = 16
            self.layeri0 = nn.Sequential(
                nn.Conv2d(3,
                          16,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False), norm_layer(self.inplanes),
                nn.ReLU(inplace=True))
            self.layeri1 = self._make_layer(BasicBlock,
                                            16,
                                            1,
                                            stride=1,
                                            )
            self.layer0 = self._make_layer(BasicBlock,
                                           32,
                                           2,
                                           stride=2,
                                           layer_conf=layer_conf['layer0']
                                           )
            norm_layer = self._norm_layer
            self.layer1 = self._make_layer(BasicBlock,
                                           64,
                                           layers[0],
                                           stride=2,
                                           layer_conf=layer_conf['layer1']
                                           )
        else:
            self.conv1 = nn.Conv2d(3,
                                   self.inplanes,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block,
                                           64,
                                           layers[0],
                                           stride=1,
                                           )
        if mulit_stage:
            if change_s1:
                self.inplanes = 16
                self.layer_i0_2x = nn.Sequential(
                    nn.Conv2d(3,
                              16,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              bias=False), norm_layer(self.inplanes),
                    nn.ReLU(inplace=True))
                self.layer_i1_2x = nn.Sequential(
                    InceptionBlock(),
                    norm_layer(16),
                    nn.ReLU(inplace=True)
                )
                self.layer0_2x = self._make_layer(BasicBlock,
                                                  32,
                                                  1,
                                                  stride=2,
                                                  layer_conf=layer_conf_conv4_conv2
                                                  )
                self.layer1_2x = self._make_layer(BasicBlock,
                                                  32,
                                                  2,
                                                  stride=2,
                                                  layer_conf=layer_conf_conv4_conv2
                                                  )
                self.layer_fuse = nn.Sequential(
                    nn.Conv2d(96,
                              64*block.expansion,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False), norm_layer(64*block.expansion),
                    nn.ReLU(inplace=True))
                self.n2_2x = L2Norm(32)
            else:
                self.inplanes = 16
                self.conv1_s = nn.Conv2d(3,
                                         self.inplanes,
                                         kernel_size=7,
                                         stride=2,
                                         padding=3,
                                         bias=False)
                self.bn1_s = norm_layer(self.inplanes)
                self.relu_s = nn.ReLU(inplace=True)
                self.maxpool_s = nn.MaxPool2d(
                    kernel_size=3, stride=2, padding=1)
                self.layer0_s = self._make_layer(block,
                                                 16,
                                                 1,
                                                 layer_conf=dict(
                                                     conv=dict(k=3, p=1, s=1, d=1), down=dict(k=1, p=0, s=1, d=1))
                                                 )
                self.layer1_s = self._make_layer(block,
                                                 16,
                                                 2,
                                                 layer_conf=dict(conv=dict(k=3, p=1, s=2, d=1),
                                                                 down=dict(k=1, p=0, s=2, d=1))
                                                 )
                self.layer_fuse = nn.Sequential(
                    nn.Conv2d((64+16)*block.expansion,
                              64*block.expansion,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False), norm_layer(64*block.expansion),
                    nn.ReLU(inplace=True))
                self.s2_2x_up=nn.Conv2d(16*block.expansion,
                              64,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
                self.n2_2x = L2Norm(64)
            if not self.classification:
                self.n2 = L2Norm(64)
                self.s2_up = nn.ConvTranspose2d(in_channels=64*block.expansion,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2,
                                                padding=1,
                                                output_padding=0,
                                                bias=False)
                self.n_cat = L2Norm(64)
                self.s_conv_s = nn.Sequential(
                    nn.Conv2d(in_channels=64+64+64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False), norm_layer(64),
                    nn.ReLU(inplace=True))
                self.hm_s = nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ), nn.Sigmoid())
                self.wh_s = nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=2,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ))
                self.offset_s = nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=2,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ))
        self.inplanes = 64*block.expansion
        self.inplanes_s2 = self.inplanes
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       layer_conf=layer_conf['layer2']
                                       )
        self.inplanes_s3 = self.inplanes
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       layer_conf=layer_conf['layer3']
                                       )
        self.inplanes_s4 = self.inplanes
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       layer_conf=layer_conf['layer4'])
        if not self.classification:
            self.inplanes_s5 = self.inplanes
            self.s3_up = nn.ConvTranspose2d(in_channels=self.inplanes_s3,
                                            out_channels=transform_planes,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            output_padding=0,
                                            bias=False)
            self.s4_up = nn.ConvTranspose2d(in_channels=self.inplanes_s4,
                                            out_channels=transform_planes,
                                            kernel_size=4,
                                            stride=4,
                                            padding=0,
                                            output_padding=0,
                                            bias=False)
            self.s5_up = nn.ConvTranspose2d(in_channels=self.inplanes_s5,
                                            out_channels=transform_planes,
                                            kernel_size=4,
                                            stride=4,
                                            padding=0,
                                            output_padding=0,
                                            bias=False)

            self.n3 = L2Norm(transform_planes)
            self.n4 = L2Norm(transform_planes)
            self.n5 = L2Norm(transform_planes)

            self.cat_up = nn.ConvTranspose2d(in_channels=transform_planes,
                                            out_channels=64,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1,
                                            output_padding=0,
                                            bias=False)
            self.s_conv = nn.Sequential(
                nn.Conv2d(in_channels=transform_planes * 3,
                        out_channels=transform_planes,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False), norm_layer(transform_planes),
                nn.ReLU(inplace=True))
            self.hm = nn.Sequential(
                nn.Conv2d(
                    in_channels=transform_planes,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ), nn.Sigmoid())
            self.wh = nn.Sequential(
                nn.Conv2d(
                    in_channels=transform_planes,
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
            self.offset = nn.Sequential(
                nn.Conv2d(
                    in_channels=transform_planes,
                    out_channels=2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    replace_with_bn=False,
                    layer_conf=dict(conv=dict(k=3, p=1, s=1, d=1), down=dict(k=1, p=0, s=1, d=1))):
        down = layer_conf['down']
        conv = layer_conf['conv']
        norm_layer = self._norm_layer
        downsample = None
        self.dilation = conv['d']
        stride=conv['s']
        if stride != 1 or self.inplanes != planes * block.expansion:
            if replace_with_bn:
                norm_layer = group_norm
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=down['k'],
                          stride=down['s'],
                          padding=down['p'],
                          bias=False,
                          dilation=down['d']),
                norm_layer(planes * block.expansion),
            )
            norm_layer = self._norm_layer
        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  self.groups,
                  self.base_width,
                  norm_layer=norm_layer,
                  layer_conv=conv,
                  replace_with_bn=replace_with_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))   

        return nn.Sequential(*layers)

    def forward(self, x):
        img = x
        # 对第一阶段进行解析, 输出64c
        if self.change_s1:
            x = self.layeri0(x) 
            x = self.layeri1(x) 
            x = self.layer0(x)
            s2 = self.layer1(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            s2 = self.layer1(x)
        if self.mulit_stage:
            x = nn.functional.interpolate(img, scale_factor=2, mode='bilinear')
            if self.change_s1:
                x = self.layer_i0_2x(x)
                x = self.layer_i1_2x(x)
                s2_2x = self.layer0_2x(x)
                s2_4x = self.layer1_2x(s2_2x)
                s2 = torch.cat([s2, s2_4x], 1)
                s2 = self.layer_fuse(s2)
            else:
                x = self.conv1_s(x)
                x = self.bn1_s(x)
                x = self.relu_s(x)
                x = self.maxpool_s(x)
                s2_2x = self.layer0_s(x)
                s2_4x = self.layer1_s(s2_2x)
                s2_2x=self.s2_2x_up(s2_2x)
                s2 = torch.cat([s2, s2_4x], 1)
                s2 = self.layer_fuse(s2)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        if self.classification:
            x = self.avgpool(s5)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            s3 = self.s3_up(s3)
            s4 = self.s4_up(s4)
            s5 = self.s5_up(s5)
            s3 = self.n3(s3)
            s4 = self.n4(s4)
            s5 = self.n5(s5)
            s_cat = torch.cat([s3, s4, s5], 1)
            s_cat = self.s_conv(s_cat)

            hm = self.hm(s_cat)
            wh = self.wh(s_cat)
            offset = self.offset(s_cat)
            return_data = dict(hm=hm, wh=wh, offset=offset)
            if self.mulit_stage:
                s_cat = self.cat_up(s_cat)
                s_cat = self.n_cat(s_cat)
                s2 = self.s2_up(s2)
                s2 = self.n2(s2)
                s2_2x = self.n2_2x(s2_2x)
                s_cat_s = torch.cat([s_cat, s2, s2_2x], 1)
                s_cat_s = self.s_conv_s(s_cat_s)
                hm_small = self.hm_s(s_cat_s)
                wh_small = self.wh_s(s_cat_s)
                offset_small = self.offset_s(s_cat_s)
                return_data.update(dict(hm_small=hm_small,
                                wh_small=wh_small, offset_small=offset_small))
            
            return return_data

    def init_weights(
            self,
            pretrained='',
    ):
        if osp.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            if 'state_dict' in pretrained_dict.keys():
                pretrained_dict = pretrained_dict['state_dict']
            elif 'model' in pretrained_dict.keys():
                pretrained_dict = pretrained_dict['model']
                new_dict = dict()
                for k, v in pretrained_dict.items():
                    new_dict[k[7:]] = v
                pretrained_dict = new_dict
        elif pretrained:
            pretrained_dict = load_state_dict_from_url(
                model_urls[pretrained], map_location='cpu')
            # self.load_state_dict(pretrained_dict)
        model_dict = self.state_dict()
        if pretrained:
            temp_dict=dict()
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and v.shape==model_dict[k].shape and 'layer2.0.conv1.weight' not in k and 'layer3.0.conv1.weight' not in k and 'layer4.0.conv1.weight' not in k :
                    temp_dict[k]=v
                else:
                    print(k)
            input('miss params')
            pretrained_dict=temp_dict
        else:
            pretrained_dict = self.state_dict()
        # for k, v in self.named_parameters():
        #     if 'layer_i0_2x' in k:
        #         break
        #     v.requires_grad = False
        for k, v in self.named_parameters():
            if 'layer2.0' in k:
                break
            v.requires_grad=False
            # if 'bn' in k and k in pretrained_dict.keys():
            print(k, v.shape)
        input('model_parameters:')
        for k, v in self.named_parameters():
            if 'hm' in k:
                if 'bias' in k:
                    pretrained_dict[k] = torch.ones_like(v) * -math.log(
                        (1 - 0.01) / 0.01)
            print(k, v.requires_grad)
        input('grad:')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        for k, v in pretrained_dict.items():
            print(k, v.shape)
        input('pretrained_parameters')


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    print(kwargs)
    # if pretrained:
    #     model.init_weights(pretrained=arch)
    print(model)
    return model


def resnet18(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',
                   BasicBlock, [2, 2, 2, 2],
                   pretrained,
                   progress,
                   **kwargs)


def resnet50(pretrained=True, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    return _resnet('resnet50',
                   Bottleneck, [3, 4, 6, 3],
                   pretrained,
                   progress,
                   **kwargs)


def test():
    import torch
    layer_conf = dict(
            layer2=dict(conv=dict(k=3, p=1, d=1, s=2),
                        down=dict(k=1, p=0, d=1, s=2)),
            layer3=dict(conv=dict(k=3, p=1, d=1, s=2),
                        down=dict(k=1, p=0, d=1, s=2)),
            layer4=dict(conv=dict(k=3, p=2, d=2, s=1),
                        down=dict(k=1, p=0, d=1, s=1)))
    layer_conf = dict(
            layer2=dict(conv=dict(k=5, p=2, d=1, s=2),
                        down=dict(k=1, p=0, d=1, s=2)),
            layer3=dict(conv=dict(k=5, p=2, d=1, s=2),
                        down=dict(k=1, p=0, d=1, s=2)),
            layer4=dict(conv=dict(k=5, p=4, d=2, s=1),
                        down=dict(k=1, p=0, d=1, s=1)))
    net = resnet18(layer_conf=layer_conf)
    net.cuda()
    for k, v in net.named_parameters():
        print(k)
    print(net)
    a = torch.ones(10, 3, 704, 704).cuda()
    while True:
        import time
        last=time.time()
        b = net(a)
        c = 0
        for name, output in b.items():
            # print(output.sum())
            c += output.mean()
            # print(name, output.shape)
        # print(c)
        c.backward()
        print(time.time()-last)
    input('s')


if __name__ == "__main__":
    test()
