import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
#from torchvision.datasets import MNIST
import os
from models import _EncoderNoa, _DecoderNoa
import time
#from tensorboardX import SummaryWriter
#import FontsDataLoaderCross
import sys
from skimage import io
import numpy as np
import cv2
from util import util
from data import CreateDataLoader
#import scipy.misc
import imageio
import glob

useConcat = True
comparedEpoch = 200


destHtmlsPath = '/mnt/data/messin/dev/python/autoEncoder/TestHtmls/Apr17/ComparedEpoch_200/'

#path4 = '/mnt/data/messin/dev/python/autoEncoder/outTest/apr14_withTripletAndCycle_Margin1_Epoch200'
path1 = '/mnt/data/messin/dev/python/autoEncoder/outTest/apr15_L1withFMLossMargin1_Epoch200'
path2 = '/mnt/data/messin/dev/python/autoEncoder/outTest/apr15_L1Margin1_TripletLossFix_Epoch200'
path3 = '/mnt/data/messin/dev/python/autoEncoder/outTest/apr14_withTripletAndCycle_L1_Epoch200'

methodDisplayNames = ['Margin1_FMLoss_L1', 'L1TripletLossFix', 'L1']

allPaths  = [path1, path2, path3]

if not os.path.exists(destHtmlsPath):
    os.makedirs(destHtmlsPath)

fid = open('%s/comparison.html'%(destHtmlsPath), 'w')

allPathsLists = []
for p in allPaths:
    imageFileList = glob.glob('%s/*.png'%p)
    imageFileList.sort()
    allPathsLists.append(imageFileList)

for i in range(0, len(allPathsLists[0]), 6):
    for j in range(len(allPathsLists)):
        fid.write('<font size="5">%s  </font>' % methodDisplayNames[j])
        fid.write('<br />')

        curList = allPathsLists[j]
        for k in range(0,6):
            f1Path = curList[i+k]
            fid.write('<img src="%s" width="160" height="160" />' % f1Path)
            fid.write('    ')

        fid.write('<hr/>')
        fid.write('<br />')
        fid.write('<br />')


    fid.write('<br />')
    fid.write('<br />')
    fid.write('<hr/>')
    fid.write('<hr/>')







