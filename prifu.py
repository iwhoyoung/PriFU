# Copyright (c) OpenMMLab. All rights reserved.
from cv2 import threshold
import torch
from torch import nn, relu
from torch.nn import Parameter

# 离散化输出
class PriFUwithBN(nn.Module):
    def __init__(
            self,
            num_features,
            affine=False,
            threshold =1e-1          
    ):
        super(PriFUwithBN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=affine)
        self.feat_filter=FeatFilter(num_features, alpha=threshold)

    def forward(self, x):
        x = self.bn(x)
        out = self.feat_filter(x)
        return out


class FeatFilter(nn.Module):
    def __init__(
            self,
            num_features,
            alpha =2e-1          
    ):
        super(FeatFilter, self).__init__()
        self.alpha = alpha
        self.num_features = num_features
        self.weight = Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = Parameter(torch.ones(1, num_features, 1, 1))
        self.mask_weight = None
        nn.init.constant_(self.weight, 0.1)
        nn.init.constant_(self.bias, 0)


    def forward(self, x):
        x, weight = RedistributeGrad.apply(x, self.weight)
        threshold = self.alpha * torch.max(weight,dim=1,keepdim=True).values
        mask = 0.5*(1+torch.sign(self.weight - threshold))
        self.mask_weight = mask * self.weight
        out = self.mask_weight * x + self.bias
        return out

class RedistributeGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):

        ctx.save_for_backward(weight)
        return x, weight

    @staticmethod
    def backward(ctx, grad_x, grad_weight):

        weight, = ctx.saved_tensors
        # set the beta
        return grad_x, grad_weight + weight * 1e-3
