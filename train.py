import sys
import os
from pathlib import Path

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import random_split

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_path', metavar='TRAIN',
                    help='path to train data folder')
parser.add_argument('test_path', metavar='TEST',
                    help='path to test data folder')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('--batch_size', '-b', metavar='BATCH', default=1,type=int,
                    help='batch size')

# parser.add_argument('gpu',metavar='GPU', type=str, default='0',
#                     help='GPU id to use.')

parser.add_argument('--task','-t', metavar='TASK', type=str, default='0',
                    help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    print(args)
    args.original_lr = 1e-7
    args.lr = 1e-7
#     args.batch_size    = 9
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    
    train_list, test_list = getTrainAndTestListFromPath(args.train_path, args.test_path)
    splitRatio = 0.8
    
    print('batch size is ', args.batch_size)
    print('cuda available? {}'.format(torch.cuda.is_available()))
    
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.to(device)
    
    criterion = nn.MSELoss(size_average=False).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        subsetTrain, subsetValid = getTrainAndValidateList(train_list, splitRatio)
        
        train(subsetTrain, model, criterion, optimizer, epoch, device)
        prec1 = validate(subsetValid, model, criterion, device)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

def getTrainAndValidateList(train_list, ratio=0.8):
    perm = np.random.permutation(len(train_list))
    subsetTrain = []
    subsetValid = []
    for i in range(len(perm)):
        index = perm[i]
        if i < ratio*len(perm):
            subsetTrain.append(train_list[index])
        else:
            subsetValid.append(train_list[index])
    return subsetTrain, subsetValid

def getTrainAndTestListFromPath(trainPath, testPath):
    trainPath = Path(trainPath)
    testPath = Path(testPath)
    print('trainPath is {}'.format(trainPath))
    assert trainPath.exists()
    assert testPath.exists()
    train_list = []
    test_list = []
    for imgPath in (trainPath / 'images').glob('*.jpg'):
        train_list.append(str(imgPath))
    for imgPath in (testPath / 'images').glob('*.jpg'):
        test_list.append(str(imgPath))
    print('train_list length is', len(train_list))
    print('test_list length is', len(test_list))
    return train_list, test_list

def train(train_list, model, criterion, optimizer, epoch, device):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=[
                       transforms.ToTensor(),
                       transforms.Resize(768),
                       transforms.RandomCrop(768),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ], 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        
        target = target.type(torch.FloatTensor).to(device)
        target = Variable(target)
        
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
def validate(val_list, model, criterion, device):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=[
                       transforms.ToTensor(),
                       transforms.Resize(768),
                       transforms.RandomCrop(768),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ],
                   batch_size=1,
                   train=False),
    batch_size=1)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
    
if __name__ == '__main__':
    main()        