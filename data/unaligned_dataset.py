
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
        self.transformFM = get_transformFMLoss(opt)
        #self.nintrm = opt.nintrm


    def __getitem__(self, index):
        #start_time = time.time()
        A_path = self.A_paths[index % self.A_size]

        prts = A_path.split('.png--')
        prts = prts[0].split('_')
        letter = prts[-1]
        letteri = ord(letter) - 65
        other_letteri = random.randint(0, 25)

        # while other_letteri == letteri:
        #     other_letteri = random.randint(0, 25)

        other_letter = letters[other_letteri]
        A_path1 = re.sub('[1,2]{1}[.]png$', '1.png', A_path)
        B_path2 = re.sub('[1,2]{1}[.]png$', '2.png', A_path)
        #B_path1 = re.sub('[A-Z]{1}[.]png', '%s.png' % other_letter, A_path1)
        B_path1 = re.sub('[A-Z]{1}[.]png', '%s.png' % letters[letteri], A_path1)
        #B_path2 = re.sub('[A-Z]{1}[.]png', '%f.png' % other_letter, A_path2)


        pair_ind = random.randint(0,9)

        prts = B_path1.split('_$_')
        prt = prts[0]
        B_path1 = prt + '_$_0_2.png'
        #test that the other letter exists and that the letters are not the same
        while not os.path.isfile(B_path1):
            other_letteri = random.randint(0, 25)
            B_path1 = re.sub('[A-Z]{1}[.]png', '%s.png' % letters[other_letteri], A_path1)

        prts = B_path1.split('_$_')
        prt = prts[0]
        B_path1 = prt + '_$_%d_2.png' % pair_ind
        while not os.path.isfile(B_path1):
            #pair_ind = random.randint(0,29)
            pair_ind = random.randint(0, 9)
            #pair_ind -= 1
            B_path1 = prt + '_$_%d_2.png' % pair_ind

        A_path2 = prt + '_$_%d_1.png' % pair_ind

        A = self.load_image(A_path1)
        B = self.load_image(B_path1)
        A2 = self.load_image(A_path2)
        B2 = self.load_image(B_path2)

        if self.opt.useFeatureMatchingLoss:
            A2_FM = self.load_imageFM(A_path2)
            B2_FM = self.load_imageFM(B_path2)
            result = {'A': A, 'B': B, 'A2': A2, 'B2': B2, 'A_paths': A_path1, 'B_paths': B_path1, 'A2_FM' : A2_FM, 'B2_FM' : B2_FM}
        else:
            result = {'A': A, 'B': B, 'A2': A2, 'B2': B2, 'A_paths': A_path1, 'B_paths': B_path1}
        return result
    #       A_path.split('')
    #       tmp = A_path.replace('.png','')
    #       prts = tmp.split('_')
    #       lettera = prts[-1]
    #       letterb = lettera

        # ff = re.sub('[A-Z]{1}[.]png', letter+'.png', f)
    #       '/home/noafish/geomst/patches/AccanthisADFStdNo2-Regular_B.png--JB_B.png__24_2.png'

    #       while letterb == lettera:
    #           if self.opt.serial_batches:
    #               index_B = index % self.B_size
    #           else:
    #               index_B = random.randint(0, self.B_size - 1)
    #           B_path = self.B_paths[index_B]
    #           tmp = B_path.replace('.png','')
    #           prts = tmp.split('_')
    #           letterb = prts[-1]

    #       #vala = ord(lettera) - 64
    #       #A2 = random.randint(0, 25)

    #       A_path_B = A_path.replace('%s.png' % lettera, '%s.png' % letterb)
    #       B_path_A = B_path.replace('%s.png' % letterb, '%s.png' % lettera)

    #       A = self.load_image(A_path)
    #       B = self.load_image(B_path)
    #       A2 = self.load_image(A_path_B)
    #       B2 = self.load_image(B_path_A)



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
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

