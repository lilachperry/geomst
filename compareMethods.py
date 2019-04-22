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
from sklearn.decomposition import PCA

useConcat = True
comparedEpoch = 110
#-----------------------noa----------------------------------------------
# contEncPath = '/home/noafish/geomst/concat/cont_enc_epoch_110.pth'
# styleEncPath = '/home/noafish/geomst/concat/stl_enc_epoch_110.pth'
# decPath = '/home/noafish/geomst/concat/dec_epoch_110.pth'
# methodName = 'Apr11_Triplet_CyclePlusCodeCycle_margin0_1_noa'
#------------------------------------------------------------------------
#-----------------------apr11_withTripletAndCycle_Margin0_1----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_1/checkpoints/cont_enc_epoch_300.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_1/checkpoints/stl_enc_epoch_300.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_1/checkpoints/dec_epoch_300.pth'
# methodName = 'apr12_withTripletAndCycle_Margin0_1'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr11_withTripletAndCycle_Margin0_2----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_2/checkpoints/cont_enc_epoch_320.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_2/checkpoints/stl_enc_epoch_320.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_2/checkpoints/dec_epoch_320.pth'
# methodName = '/apr14_withTripletAndCycle_Margin0_2'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr12_withTripletAndCycle_Margin0_5----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr12_withTripletAndCycle_Margin0_5/checkpoints/cont_enc_epoch_110.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr12_withTripletAndCycle_Margin0_5/checkpoints/stl_enc_epoch_110.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr12_withTripletAndCycle_Margin0_5/checkpoints/dec_epoch_110.pth'
# methodName = '/apr12_withTripletAndCycle_Margin0_5'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr11_withTripletAndCycle_Margin0_1_FMLoss----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_1_FMLoss/checkpoints/cont_enc_epoch_110.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_1_FMLoss/checkpoints/stl_enc_epoch_110.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr11_withTripletAndCycle_Margin0_1_FMLoss/checkpoints/dec_epoch_110.pth'
# methodName = 'apr11_withTripletAndCycle_Margin0_1_FMLoss'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr14_withTripletAndCycle_L1----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr14_withTripletAndCycle_L1/checkpoints/cont_enc_epoch_200.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr14_withTripletAndCycle_L1/checkpoints/stl_enc_epoch_200.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr14_withTripletAndCycle_L1/checkpoints/dec_epoch_200.pth'
# methodName = 'apr14_withTripletAndCycle_L1'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr14_withTripletAndCycle_Margin1----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr14_withTripletAndCycle_Margin1/checkpoints/cont_enc_epoch_130.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr14_withTripletAndCycle_Margin1/checkpoints/stl_enc_epoch_130.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr14_withTripletAndCycle_Margin1/checkpoints/dec_epoch_130.pth'
# methodName = 'apr14_withTripletAndCycle_Margin1'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr15_L1withFMLossMargin1----------------------------------------------------------------------------
contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1withFMLossMargin1/checkpoints/cont_enc_epoch_110.pth'
styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1withFMLossMargin1/checkpoints/stl_enc_epoch_110.pth'
decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1withFMLossMargin1/checkpoints/dec_epoch_110.pth'
methodName = 'apr15_L1withFMLossMargin1'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr15_L1Margin1_TripletLossFix----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1Margin1_TripletLossFix/checkpoints/cont_enc_epoch_200.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1Margin1_TripletLossFix/checkpoints/stl_enc_epoch_200.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1Margin1_TripletLossFix/checkpoints/dec_epoch_200.pth'
# methodName = 'apr15_L1Margin1_TripletLossFix'
#--------------------------------------------------------------------------------------------------------------------------------------
#-----------------------apr21_L1Margin1_FMLayer8_Par9_afterConv----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer8_Par9_afterConv/checkpoints/cont_enc_epoch_110.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer8_Par9_afterConv/checkpoints/stl_enc_epoch_110.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer8_Par9_afterConv/checkpoints/dec_epoch_110.pth'
# methodName = 'apr21_L1Margin1_FMLayer8_Par9_afterConv'
#--------------------------------------------------------------------------------------------------------------------------------------
#-----------------------apr21_L1Margin1_FMLayer6_Par7----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer6_Par7/checkpoints/cont_enc_epoch_110.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer6_Par7/checkpoints/stl_enc_epoch_110.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer6_Par7/checkpoints/dec_epoch_110.pth'
# methodName = 'apr21_L1Margin1_FMLayer6_Par7'
#--------------------------------------------------------------------------------------------------------------------------------------

#-----------------------apr21_L1Margin1_FMLayer9_Par10----------------------------------------------------------------------------
# contEncPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer9_Par10/checkpoints/cont_enc_epoch_110.pth'
# styleEncPath = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer9_Par10/checkpoints/stl_enc_epoch_110.pth'
# decPath =  '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr21_L1Margin1_FMLayer9_Par10/checkpoints/dec_epoch_110.pth'
# methodName = 'apr21_L1Margin1_FMLayer9_Par10'
#--------------------------------------------------------------------------------------------------------------------------------------


destBasePath = '/mnt/data/messin/dev/python/autoEncoder/outTest/'
destPath = '%s/%s_Epoch%d'%(destBasePath, methodName, comparedEpoch)

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
#parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the low resolution image size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--MSEWeight', type=int, default=100, help='the weight of the MSE loss')
parser.add_argument('--learningRate', type=float, default=0.001, help='learning rate for optimizer')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
parser.add_argument('--indir', type=str, default='', help='folder to input model checkpoints')

opt = parser.parse_args()
print(opt)

opt.phase = 'test'
opt.dataset_mode = 'unaligned'
opt.dataroot = '/mnt/data/noafish/geomst/datasets/fonts'
opt.resize_or_crop = 'resize_and_crop'
opt.fineSize = opt.imageSize #64
opt.loadSize = opt.imageSize #64
opt.isTrain = False
opt.no_flip = True
opt.gray = False
opt.serial_batches = False
opt.nThreads = 4
opt.max_dataset_size = float("inf")
#opt.which_direction = 'AToB'
opt.input_nc = 1
opt.nc = 1

def runMethodTest(destPath, opt, contEncPath, styleEncPath, decPath, useConcat):
	if not os.path.exists(destPath):
		os.makedirs(destPath)

	data_loader = CreateDataLoader(opt)
	dataloader = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#testing images = %d' % dataset_size)

	cont_enc = _EncoderNoa(opt.imageSize)
	stl_enc = _EncoderNoa(opt.imageSize)
	dec = _DecoderNoa(opt.imageSize, useConcat)
	cont_enc.load_state_dict(torch.load(contEncPath))
	stl_enc.load_state_dict(torch.load(styleEncPath))
	dec.load_state_dict(torch.load(decPath))

	if opt.cuda:
		# feature_extractor.cuda()
		cont_enc.cuda()
		stl_enc.cuda()
		dec.cuda()

	torch.manual_seed(0)

	for i, data in enumerate(dataloader, 0):

		img1 = data['A']
		img2 = data['B']
		img12 = data['A2']
		img21 = data['B2']

		if opt.cuda:
			img1 = img1.cuda()
			img2 = img2.cuda()
			img12 = img12.cuda()
			img21 = img21.cuda()

		stl1 = stl_enc(img1)
		stl2 = stl_enc(img2)
		cont1 = cont_enc(img1)
		cont2 = cont_enc(img2)

		# stl12 = stl_enc(img12)
		# stl21 = stl_enc(img21)
		# cont12 = cont_enc(img12)
		# cont21 = cont_enc(img21)

		if (useConcat):
			stl1cont2 = torch.cat((stl1, cont2), 1)
			stl2cont1 = torch.cat((stl2, cont1), 1)
			# stl1cont1 = torch.cat((stl1, cont1), 1)
			# stl2cont2 = torch.cat((stl2, cont2), 1)
		else:
			stl1cont2 = stl1 + cont2
			stl2cont1 = stl2 + cont1
			# stl1cont1 = stl1 + cont1
			# stl2cont2 = stl2 + cont2

		dec12 = dec(stl1cont2)
		dec21 = dec(stl2cont1)
		# dec11 = dec(stl1cont1)
		# dec22 = dec(stl2cont2)

		if i % 10 == 0:
			im1 = util.tensor2im(img1[0])
			im2 = util.tensor2im(img2[0])
			oim12 = util.tensor2im(img12[0])
			oim21 = util.tensor2im(img21[0])
			im12 = util.tensor2im(dec12[0])
			im21 = util.tensor2im(dec21[0])

			imageio.imwrite(os.path.join(destPath, '%d_style_1_cont_2.png' % (i)), im12)
			imageio.imwrite(os.path.join(destPath, '%d_style_2_cont_1.png' % (i)), im21)
			imageio.imwrite(os.path.join(destPath, '%d_style_1_cont_1_orig.png' % (i)), im1)
			imageio.imwrite(os.path.join(destPath, '%d_style_2_cont_2_orig.png' % (i)), im2)
			imageio.imwrite(os.path.join(destPath, '%d_style_1_cont_2_orig.png' % (i)), oim12)
			imageio.imwrite(os.path.join(destPath, '%d_style_2_cont_1_orig.png' % (i)), oim21)


runMethodTest(destPath, opt, contEncPath, styleEncPath, decPath, useConcat)
#nclasses = 26
#nClassesFonts = 138
#useLetterClassificationLoss = False

#modelPath = '/home/messin/dev/python/Pytorch-srgan/fontClassifier/trainedModels/trainFontClassifier_may2_uniqueSubset132_b32_class1_4/checkpoints/classifier_ep199.pth'
#letterClassificationModelPath = '/mnt/data/messin/dev/python/letterClassifier/outTrain/Feb4_b32_26classFix/checkpoints/classifier_ep23.pth'

#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/train_spirax_batch#16_imSize#28_Jan20_RefWeight50To200/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/trainfullLetter_Jan29_batch32_imSize28_Normalized_sigmoidInsteadOfTanh_MSEandFM/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/toDelete/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/trainCross_Apr2_normFix_cycleLoss/'














