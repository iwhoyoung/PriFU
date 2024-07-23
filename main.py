import os

import torchvision
from torchvision.datasets import CIFAR10
from cifar2 import CIFAR2
from resnet import ResNet, BasicBlock
from prifu import PriFUwithBN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from PIL import Image
import numpy as np

import torch
import time
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import pandas as pd
import torch.nn.functional as F

# hyper-parameter
binary_num = 64
alpha = 1
beta = 1
gamma = 1
use_cuda = True
nThreads = 2
lr = 0.1
server_mode = True
epoch_num = 80

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def train(train_loader, model, criterion, optimizer,mean_loss=100):
    model.train()
    train_loss = []
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        classes = model(input)
        loss = criterion(classes, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        train_loss.append(loss.item())      
    return np.mean(train_loss)


def predict(test_loader, model):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [losses, top1, top5],
        prefix='Test: ')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class NoiseInput(nn.Module):
    def __init__(
            self,       
    ):
        super(NoiseInput, self).__init__()

    def forward(self, x):
        size = x.size()
        ones = torch.ones((1,size[1],size[2]),requires_grad=False).to(x.device)
        uniforms = torch.rand((1,size[1],size[2]),requires_grad=False).to(x.device)
        norms = torch.clamp(0.5+torch.randn((1,size[1],size[2]),requires_grad=False).to(x.device),min=0,max=1)
        out = torch.cat([x,ones,uniforms,norms],dim=0)
        return out

if __name__ == '__main__':
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, norm_layer=PriFUwithBN)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=0)
    # optimizer = torch.optim.SGD(model.fc_privacy.parameters(), lr, weight_decay=0)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9,
    #                             weight_decay=0.0005)
    best_loss = 10000.0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    transforms_c = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # NoiseInput()
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # NoiseInput()
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    if use_cuda:
        model = model.cuda()
    # choose 2-classification or 10-classification
    # train_data = CIFAR2('/asset/dataset/cifar10/', train=True, transform=transforms_c)
    # test_data = CIFAR2('/asset/dataset/cifar10/', train=False, transform=transforms_test)
    train_data = CIFAR10('/home/hy/asset/dataset/cifar10/', train=True, transform=transforms_c)
    test_data = CIFAR10('/home/hy/asset/dataset/cifar10/', train=False, transform=transforms_test)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    train_loss=100.
    for epoch in range(epoch_num):
        train_loss = train(train_loader, model, criterion, optimizer,train_loss)
        if epoch % 1 == 0:
            acc = predict(test_loader, model)
            # torch.save(model.state_dict(), '/model/model_param.pth')
        scheduler.step() 
    acc = predict(test_loader, model)
