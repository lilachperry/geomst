
import os.path
from data.base_dataset import BaseDataset, get_transform, get_transformFMLoss
from data.image_folder import make_dataset
from PIL import Image
import random
import time
import ntpath
import numpy as np
import torch
import re

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        if opt.nc == 1:
            opt.gray = True
        self.opt = opt
        #self.root = opt.dataroot
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dirA= os.path.join(opt.datarootA, opt.phase)
        self.dirB = os.path.join(opt.datarootB, opt.phase)

        self.APaths = make_dataset(self.dirA)
        self.BPaths = make_dataset(self.dirB)

        self.APaths = sorted(self.APaths)
        self.BPaths = sorted(self.BPaths)
        self.ASize = len(self.APaths)
        self.BSize = len(self.BPaths)
        self.aName = self.opt.aName
        self.bName = self.opt.bName
        self.transform = get_transform(opt)
        self.transformFM = get_transformFMLoss(opt)
        #self.nintrm = opt.nintrm


    def __getitem__(self, index):
        #start_time = time.time()
        APath = self.APaths[index % self.ASize]
        BPath = self.BPaths[index % self.BSize]

        contentAStyleBPath = APath.replace(self.aName, self.bName)
        contentBStyleAPath = BPath.replace(self.bName, self.aName)

        A = self.load_image(APath)
        B = self.load_image(BPath)
        contentAStyleB = self.load_image(contentAStyleBPath)
        contentBStyleA = self.load_image(contentBStyleAPath)

        if self.opt.useFeatureMatchingLoss:
            contentAStyleB_FM = self.load_imageFM(contentAStyleBPath)
            contentBStyleA_FM = self.load_imageFM(contentBStyleAPath)
            result = {'A': A, 'B': B, 'contentAStyleB': contentAStyleB, 'contentBStyleA': contentBStyleA, 'APath': APath, 'BPath': BPath, 'contentAStyleB_FM' : contentAStyleB_FM, 'contentBStyleA_FM' : contentBStyleA_FM}
        else:
            result = {'A': A, 'B': B, 'contentAStyleB': contentAStyleB, 'contentBStyleA': contentBStyleA, 'APath': APath, 'BPath': BPath}
        return result

    #@profile
    def load_image(self, im_path):
        A_img = Image.open(im_path).convert('RGB')
        sz = A_img.size
        self.orig_size = sz
        A = self.transform(A_img)

        #if self.opt.nc == 1:  # RGB to gray
        #    tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #    A = tmp.unsqueeze(0)
        return A
        #return {'img': A, 'dims': sz}
        #return A, sz

    def load_imageFM(self, im_path):
        A_img = Image.open(im_path).convert('RGB')
        sz = A_img.size
        self.orig_size = sz
        A = self.transformFM(A_img)

        #if self.opt.nc == 1:  # RGB to gray
        #    tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #    A = tmp.unsqueeze(0)
        return A

    def __len__(self):
        return max(self.ASize, self.BSize)

    def name(self):
        return 'UnalignedDataset'

