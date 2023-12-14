import torch
from torch import nn
from torch.nn import functional as F 
import math

from GCNet.Extractor import GC_Extractor
from GCNet.EncoderDecoder import Hourglass


class GCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea1 = nn.Sequential( GC_Extractor(3, 32, 8) )
        self.hourglass = nn.Sequential( Hourglass() )

    def forward(self, imgL, imgR, min_disp, max_disp):
        #extract feature map
        featureL = self.fea1(imgL) 
        featureR = self.fea1(imgR) # shape -> C * H/2 * W/2

        # construct cost volume
        cost_vol = self.cost_volume(featureL, featureR, min_disp, max_disp) # shape -> B * 2C * (maxdisp-mindisp)/2 * H/2 * W/2

        # cost filtering
        cost_vol = self.hourglass(cost_vol) # shape -> B * 1 * (maxdisp-mindisp) * H * W

        # disparity regression
        disp = self.softargmax(cost_vol, min_disp, max_disp) # shape -> B * H * W
        return disp

    def cost_volume(self, feaL:torch.tensor, feaR:torch.tensor, min_disp, max_disp) -> torch.tensor:
        B, C, H, W = feaL.shape
        device = feaL.device

        # feature map has been downsample, so disparity range should be devided by 2
        max_disp = int(max_disp/2)
        min_disp = int(min_disp/2)
        cost = torch.zeros(B, C*2, max_disp-min_disp, H, W).to(device)
        # cost[:, 0:C, :, :, :] = feaL.unsqueeze(2).repeat(1,1,max_disp-min_disp,1,1)


        for i in range(min_disp, max_disp):
            if i < 0:
                cost[:, 0:C, i, :, 0:W+i] = feaL[:, :, :, 0:W+i]
                cost[:, C:, i, :, 0:W+i] = feaR[:, :, :, -i:]
            if i >= 0:
                cost[:, 0:C, i, :, i:] = feaR[:, :, :, i:]
                cost[:, C:, i, :, i:] = feaR[:, :, :, :W-i]
        return cost
    
    def softargmax(self, cost, min_disp, max_disp):
        cost_softmax = F.softmax(cost, dim = 2)
        vec = torch.arange(min_disp, max_disp).to(cost.device)
        vec = vec.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        vec = vec.expand_as(cost_softmax).type_as(cost_softmax)
        disp = torch.sum(vec*cost_softmax, dim=2)
        return disp
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()