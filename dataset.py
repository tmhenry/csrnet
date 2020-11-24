import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F
import cv2

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883
        
        if self.transform is not None:
            # resize and randomCrop when batch size is not 1
            if self.batch_size > 1:
                for i in range(len(self.transform)):
                    t = self.transform[i]
                    img = t(img)
                    # transform target only for resize and randomCrop
                    if i != 3:
                        target = t(target)
                    else:
                        target = F.resize(target, 96)
            else:
                for i in range(len(self.transform)):
                    # pass all the resize and randomCrop
                    if i != 1 and i != 2:
                        img = t(img)
                target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
                target = F.to_tensor(target)

        return img,target