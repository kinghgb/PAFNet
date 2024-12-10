import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys
from mmcv.cnn import build_norm_layer
sys.path.insert(0, '../../')
from mmcv.ops.carafe import CARAFEPack
from pvtv2 import pvt_v2_b0,pvt_v2_b1,pvt_v2_b2,pvt_v2_b2_li,pvt_v2_b3,pvt_v2_b4,pvt_v2_b5
affine_par = True
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class SARNet(nn.Module):
    def __init__(self, fun_str = 'pvt_v2_b3'):
        super().__init__()
        self.backbone,embedding_dims = eval(fun_str)()
        self.easpp = GPM()
        self.clff1 = CLFF1()
        self.clff2 = CLFF2()
        self.clff3 = CLFF3()
        self.fgc0 = FGC(embedding_dims[0] // 4, embedding_dims[0] // 2, focus_background=False,
                                     opr_kernel_size=7, iterations=1)
        self.erm3 = ERM(embedding_dims[3] // 8, embedding_dims[3] // 8,
                                     opr_kernel_size=7, iterations=1)
        self.erm2 = ERM(embedding_dims[1]//2, embedding_dims[3] // 8,
                                     opr_kernel_size=7, iterations=1)
        self.erm1 = ERM(embedding_dims[0]//2, embedding_dims[1] // 2,
                                     opr_kernel_size=7,
                                     iterations=1)
        self.erm0 = ERM(embedding_dims[0] // 4, embedding_dims[0] // 2,
                                     opr_kernel_size=7, iterations=1)
        # self.oaa0 = OAA(cur_in_channels=embedding_dims[0], low_in_channels=embedding_dims[1],
        #                           out_channels=embedding_dims[0]//2, cur_scale=1, low_scale=2)
        # self.oaa1 = OAA(cur_in_channels=embedding_dims[1], low_in_channels=embedding_dims[2],
        #                           out_channels=embedding_dims[1]//2, cur_scale=1, low_scale=2)
        # self.oaa2 = OAA(cur_in_channels=embedding_dims[2], low_in_channels=embedding_dims[3],
        #                           out_channels=embedding_dims[3] // 8, cur_scale=1, low_scale=2)

        self.cbr = CBR(in_channels=embedding_dims[3], out_channels=embedding_dims[3] // 8,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)


        self.predict_conv = nn.Sequential(nn.Conv2d(in_channels=embedding_dims[3] // 8, out_channels=1, kernel_size=3, padding=1, stride=1))

        self.oaa3 = OAA1()  # 16   #out_channels=embedding_dims[0]//4
        self.predict_conv4 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=1))
        self.predict_conv3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1))
        self.predict_conv2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1))
        self.predict_conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1))
        self.predict_conv0 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1))
#模型
    def forward(self, x):
        # byxhz

        layer = self.backbone(x)
        s5 = self.cbr(layer[3])
        s4 = self.clff1(layer[2], layer[3],s5)
        s3 = self.clff2(layer[1], layer[2],s4)
        s2 = self.clff3(layer[0], layer[1],s3)

        s1 = self.oaa3(s2)

        predict4 = self.easpp(layer[3])
        # focus
        fgc3, predict3 = self.erm3(s4, s5, predict4)

        fgc2, predict2 = self.erm2(s3, fgc3, predict3)

        fgc1, predict1 = self.erm1(s2, fgc2, predict2)

        fgc0, predict0 = self.fgc0(s1, fgc1, predict1)

        # rescale
    
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)

        return predict4, predict3, predict2, predict1, predict0
class SENet(nn.Module):

    def __init__(self, in_dim, ratio=2):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // ratio, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.dim // ratio, in_dim, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        y = self.sigmoid(self.fc(y)).view(b, c, 1, 1)

        output = y.expand_as(x)

        return output
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x
"""
    Position Attention Module (PAM)
"""


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out
# e-sapp
class GPM(nn.Module):
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=128):
        # def __init__(self, dilation_series=[2, 5, 7], padding_series=[2, 5, 7], depth=128):
        super(GPM, self).__init__()
        self.se = SENet(128)
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(512, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(512, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(512, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(512, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(512, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth*5, 256, kernel_size=3, padding=1),
            PAM(256)
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=affine_par),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = self.se(branch_main)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch0=self.se(branch0)
        branch1 = self.branch1(x)
        branch1 = self.se(branch1)
        branch2 = self.branch2(x)
        branch2 = self.se(branch2)
        branch3 = self.branch3(x)
        branch3 = self.se(branch3)
        # x1=branch0*branch1
        # x2=branch1*branch2
        # x3=branch2*branch3
        # y2=x2*x1
        # z2=x2*x3
        # y3=y2*z2
        # out=branch0+x1+y2+y3
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x


class SENet(nn.Module):

    def __init__(self, in_dim, ratio=2):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // ratio, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.dim // ratio, in_dim, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        y = self.sigmoid(self.fc(y)).view(b, c, 1, 1)

        output = y.expand_as(x)

        return output
class Fuse(nn.Module):

    def __init__(self):
        super(Fuse, self).__init__()
        self.se = SENet(64)

    def forward(self, x0, x1, x2):  # x1 and x2 are the adjacent features
        x1_up = F.upsample(x1, size=x0.size()[2:], mode='bilinear')
        x2_up = F.upsample(x2, size=x0.size()[2:], mode='bilinear')
        x_fuse = x0 + x1_up + x2_up
        x_w = self.se(x_fuse)
        output = x0 * x_w + x0

        return output


class CLFF1(nn.Module):

    def __init__(self):
        super(CLFF1, self).__init__()
        self.fuse1 = Fuse()
        self.conv1 = nn.Sequential(nn.Conv2d(64 * 3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.con1 = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.con2 = nn.Sequential(nn.Conv2d(320, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.con3 = nn.Sequential(nn.Conv2d(64, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
    def forward(self, x0, x1, x2):  # x1 and x2 are the adjacent features
        pixel_shuffle = torch.nn.PixelShuffle(2)
        x0=self.con2(x0)
        x1= self.con1(x1)
        x_g3 = self.fuse1(x0, x1, x2)
        x_g4 = self.fuse1(x1, x0, x2)
        x_g4=self.con3(x_g4)
        x_g4 = pixel_shuffle(x_g4)
        x_g5 = self.fuse1(x2, x0, x1)
        x_g5 = self.con3(x_g5)
        x_g5 = pixel_shuffle(x_g5)
        x1 = x_g3 * x_g4
        x2 = x_g4 * x_g5
        x3 = x1 * x2
        #output = x_g3 + x1 + x3
        output = self.conv1(torch.cat([x_g3, x_g4, x_g5], dim=1))

        return output

class CLFF2(nn.Module):

        def __init__(self):
            super(CLFF2, self).__init__()
            self.fuse1 = Fuse()
            self.conv1 = nn.Sequential(nn.Conv2d(64 * 3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

            self.con1 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.con2 = nn.Sequential(nn.Conv2d(320, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

            self.con3 = nn.Sequential(nn.Conv2d(64, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        def forward(self, x0, x1, x2):  # x1 and x2 are the adjacent features
            pixel_shuffle = torch.nn.PixelShuffle(2)
            x0 = self.con1(x0)
            x1 = self.con2(x1)
            x_g3 = self.fuse1(x0, x1, x2)
            x_g4 = self.fuse1(x1, x0, x2)
            x_g4 = self.con3(x_g4)
            x_g4 = pixel_shuffle(x_g4)
            x_g5 = self.fuse1(x2, x0, x1)
            x_g5 = self.con3(x_g5)
            x_g5 = pixel_shuffle(x_g5)
            x1 = x_g3 * x_g4
            x2 = x_g4 * x_g5
            x3 = x1 * x2
            #output = x_g3 + x1 + x3
            output = self.conv1(torch.cat([x_g3, x_g4, x_g5], dim=1))

            return output


class CLFF3(nn.Module):

    def __init__(self):
        super(CLFF3, self).__init__()
        self.fuse1 = Fuse()
        self.conv1 = nn.Sequential(nn.Conv2d(64*3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.con1 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.con3 = nn.Sequential(nn.Conv2d(64, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, x0, x1, x2):  # x1 and x2 are the adjacent features
        pixel_shuffle = torch.nn.PixelShuffle(2)
        x1 = self.con1(x1)
        x_g3 = self.fuse1(x0, x1, x2)
        x_g4 = self.fuse1(x1, x0, x2)
        x_g4 = self.con3(x_g4)
        x_g4 = pixel_shuffle(x_g4)
        x_g5 = self.fuse1(x2, x0, x1)
        x_g5 = self.con3(x_g5)
        x_g5 = pixel_shuffle(x_g5)
        x1=x_g3*x_g4
        x2=x_g4*x_g5
        x3=x1*x2
        # output=x_g3+x1+x3
        # output = self.conv1(output)
        output = self.conv1(torch.cat([x_g3, x_g4, x_g5], dim=1))
        return output

class OAA1(nn.Module):
    def __init__(self):
        super(OAA1,self).__init__()
        self.cur_conv = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
    def forward(self,x_cur):
        pixel_shuffle = torch.nn.PixelShuffle(2)
        x_cur=self.cur_conv(x_cur)
        x_cur = pixel_shuffle(x_cur)
        #bicubic bilinear nearest
        x = self.out_conv(x_cur)
        return x

import numpy as np
import cv2

def get_open_map(input,kernel_size,iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()
def erosion(input,kernel_size,iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    erosion = map(lambda i: cv2.erode(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(erosion)))
    return open_map_tensor.unsqueeze(1).cuda()
class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FGC(nn.Module):
    def __init__(self, channel1, channel2,focus_background = True, opr_kernel_size = 3,iterations = 1):
        super(FGC, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(2 * self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations


    def forward(self, cur_x, dep_x, in_map):
        # x; current-level features     cur_x
        # y: higher-level features    dep_x
        # in_map: higher-level prediction

        dep_x = self.up(dep_x)

        input_map = self.input_map(in_map)

        if self.focus_background:
            self.increase_map = self.increase_input_map(get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map)
            b_feature = cur_x * self.increase_map #当前层中,关注深层部分没有关注的部分

        else:
            b_feature = cur_x * input_map  #在当前层中，对深层部分关注的部分更加关注，同时也关注一下其他部分
        #b_feature = cur_x
        fn = self.conv2(b_feature)


        refine2 = self.conv_cur_dep1(torch.cat((dep_x, self.beta * fn),dim=1))
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map
class ERM(nn.Module):
    def __init__(self, channel1, channel2,opr_kernel_size = 3,iterations = 1):
        super(ERM, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        #self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(2 * self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations


    def forward(self, cur_x, dep_x, in_map):
        # x; current-level features     cur_x
        # y: higher-level features    dep_x
        # in_map: higher-level prediction

        dep_x = self.up(dep_x)#fi+1
        in_map1=1-in_map
        input_map = self.input_map(in_map)#pi+1
        input_map1 = self.input_map(in_map1)


        self.increase_map = get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map
        b_feature = cur_x * self.increase_map #当前层中,关注深层部分没有关注的部分
        self.increase_map1 = get_open_map(input_map1, self.opr_kernel_size, self.iterations) - input_map1
        a_feature =cur_x*self.increase_map1
        c_feature=b_feature+a_feature
        fn = self.conv2(c_feature)


        refine2 = self.conv_cur_dep1(torch.cat((dep_x, self.beta * fn),dim=1))
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map
if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from thop import profile
    net = SARNet('pvt_v2_b3').cuda()
    data = torch.randn(1, 3, 672, 672).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))
    y = net(data)
    for i in y:
        print(i.shape)