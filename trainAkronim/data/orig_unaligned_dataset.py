import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import time
import ntpath
import numpy as np
import torch


class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        if opt.nc == 1:
            opt.gray = True
        self.opt = opt
        self.root = opt.dataroot
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.dir_B = self.dir_A

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        #self.nintrm = opt.nintrm

    #@profile
    def __getitem__(self, index):
        #start_time = time.time()
        A_path = self.A_paths[index % self.A_size]
        tmp = A_path.replace('.png','')
        prts = tmp.split('_')
        lettera = prts[-1]
        letterb = lettera

        while letterb == lettera:
            if self.opt.serial_batches:
                index_B = index % self.B_size
            else:
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            tmp = B_path.replace('.png','')
            prts = tmp.split('_')
            letterb = prts[-1]

        #vala = ord(lettera) - 64
        #A2 = random.randint(0, 25)

        A_path_B = A_path.replace('%s.png' % lettera, '%s.png' % letterb)
        B_path_A = B_path.replace('%s.png' % letterb, '%s.png' % lettera)

        A = self.load_image(A_path)
        B = self.load_image(B_path)
        A2 = self.load_image(A_path_B)
        B2 = self.load_image(B_path_A)
        
        return {'A': A, 'B': B, 'A2': A2, 'B2': B2,
                'A_paths': A_path, 'B_paths': B_path}
                 #'reals': reals}


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


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

