
import os.path
from data.base_dataset import BaseDataset, get_transform, get_transformFMLoss
from data.image_folder import make_dataset
from PIL import Image
import PIL.ImageOps
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
        self.dirContent = opt.datarootContent
        self.dirStyle = opt.datarootStyle

        self.contentPaths = make_dataset(self.dirContent)
        self.stylePaths = make_dataset(self.dirStyle)

        self.contentPaths = sorted(self.contentPaths)
        self.stylePaths = sorted(self.stylePaths)
        self.contentSize = len(self.contentPaths)
        self.styleSize = len(self.stylePaths)
        self.transform = get_transform(opt)



    def __getitem__(self, index):
        #start_time = time.time()
        contentPath = self.contentPaths[index % self.contentSize]
        stylePath = self.stylePaths[index % self.styleSize]

        contentIm = self.load_image(contentPath, True)
        styleIm = self.load_image(stylePath, False)


        return {'content': contentIm, 'style': styleIm, 'A_paths': contentPath, 'B_paths': stylePath}



    #@profile
    def load_image(self, im_path, isContent):
        A_img = Image.open(im_path).convert('RGB')
        if self.opt.invertColorsContent and isContent:
            A_img = PIL.ImageOps.invert(A_img)
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
        return self.contentSize

    def name(self):
        return 'UnalignedDataset'

