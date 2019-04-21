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
from PIL import Image



pca = PCA(n_components=3)


def my_mse(a, b):
	d = a - b
	d = torch.mean(d ** 2)
	return d


def tensor2numpy(tensor):#, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    tensor = tensor.detach()
    sz = tensor.size()
    if len(sz) == 4:
        tensor = tensor[0]
    tensor_numpy = tensor.cpu().float().numpy()
    #image_numpy = image_tensor[0].cpu().float().numpy()
    #tensor_numpy = np.transpose(tensor_numpy, (1, 2, 0))
    #return image_numpy.astype(imtype)
    return tensor_numpy


def pca_feat_map(fm):
	x = tensor2numpy(fm)
	#print(x.shape)
	nsamples, nx, ny = x.shape
	x2 = x.reshape((nsamples,nx*ny))
	x2 = np.transpose(x2, (1, 0))
	#print(x2.shape)
	prin_comps = pca.fit_transform(x2)
	pc = prin_comps.reshape((nx,ny,3))
	minval = pc.min()
	maxval = pc.max()
	#print(minval)
	#print(maxval)
	pc2 = 255 * (pc - minval) / (maxval - minval)
	#pc2 = pc2.round()
	pc2 = pc2.astype(np.uint8)
	#minval = pc2.min()
	#maxval = pc2.max()
	#print(minval)
	#print(maxval)
	return pc2


def save_image(odir, itr, tp, stl, cont, num, img):
	#print(img.shape)
	#img2 = img.resize((64,64), Image.ANTIALIAS)
	img2 = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
	imageio.imwrite(os.path.join(opt.out,'%d_%s_%d_%d_%.2d.png' % (itr, tp, stl, cont, num)), img2)


def bla(stl_enc, cont_enc, img, stl, cont, itr):
	lst1 = stl_enc.encoder.children()
	x = img
	for idx,l in enumerate(lst1):
		x = l(x)
		#print(x.size())
		if idx == 1:
			feats1 = x
		if idx == 3:
			feats3 = x
		if idx == 7:
			feats7 = x
		if idx == 10:
			feats10 = x

	spc1 = pca_feat_map(feats1[0])
	spc3 = pca_feat_map(feats3[0])
	spc7 = pca_feat_map(feats7[0])
	spc10 = pca_feat_map(feats10[0])


	lst2 = cont_enc.encoder.children()
	x = img
	for idx,l in enumerate(lst2):
		x = l(x)
		#print(x.size())
		if idx == 1:
			feats1 = x
		if idx == 3:
			feats3 = x
		if idx == 7:
			feats7 = x
		if idx == 10:
			feats10 = x

	cpc1 = pca_feat_map(feats1[0])
	cpc3 = pca_feat_map(feats3[0])
	cpc7 = pca_feat_map(feats7[0])
	cpc10 = pca_feat_map(feats10[0])

	save_image(opt.out, itr, 'stl', stl, cont, 1, spc1)
	save_image(opt.out, itr, 'stl', stl, cont, 3, spc3)
	save_image(opt.out, itr, 'stl', stl, cont, 7, spc7)
	save_image(opt.out, itr, 'stl', stl, cont, 10, spc10)
	
	save_image(opt.out, itr, 'cont', stl, cont, 1, cpc1)
	save_image(opt.out, itr, 'cont', stl, cont, 3, cpc3)
	save_image(opt.out, itr, 'cont', stl, cont, 7, cpc7)
	save_image(opt.out, itr, 'cont', stl, cont, 10, cpc10)



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

nclasses = 26
#nClassesFonts = 138
#useLetterClassificationLoss = False

#modelPath = '/home/messin/dev/python/Pytorch-srgan/fontClassifier/trainedModels/trainFontClassifier_may2_uniqueSubset132_b32_class1_4/checkpoints/classifier_ep199.pth'
#letterClassificationModelPath = '/mnt/data/messin/dev/python/letterClassifier/outTrain/Feb4_b32_26classFix/checkpoints/classifier_ep23.pth'

#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/train_spirax_batch#16_imSize#28_Jan20_RefWeight50To200/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/trainfullLetter_Jan29_batch32_imSize28_Normalized_sigmoidInsteadOfTanh_MSEandFM/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/toDelete/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/trainCross_Apr2_normFix_cycleLoss/'


if not os.path.exists(opt.out):
	os.makedirs(opt.out)


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
data_loader = CreateDataLoader(opt)
dataloader = data_loader.load_data()
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

cont_enc_path = os.path.join(opt.indir, 'cont_enc_latest.pth')
stl_enc_path = os.path.join(opt.indir, 'stl_enc_latest.pth')
dec_path = os.path.join(opt.indir, 'dec_latest.pth')

#lfile = os.path.join(opt.out, 'loss_log.txt')
#lfile_handle = open(lfile, 'w')


cont_enc = _EncoderNoa(opt.imageSize)
stl_enc = _EncoderNoa(opt.imageSize)
dec = _DecoderNoa(opt.imageSize)
cont_enc.load_state_dict(torch.load(cont_enc_path))
stl_enc.load_state_dict(torch.load(stl_enc_path))
dec.load_state_dict(torch.load(dec_path))
#decr = _DecoderNoa(opt.imageSize)
#mse_criterion = nn.MSELoss()
#class_criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

#optimizer_cont_enc = torch.optim.Adam(cont_enc.parameters(), lr=opt.learningRate, weight_decay=1e-5)
#optimizer_stl_enc = torch.optim.Adam(stl_enc.parameters(), lr=opt.learningRate, weight_decay=1e-5)
#optimizer_dec = torch.optim.Adam(dec.parameters(), lr=opt.learningRate, weight_decay=1e-5)
#optimizer_decr = torch.optim.Adam(decr.parameters(), lr=opt.learningRate, weight_decay=1e-5)

if opt.cuda:
	#feature_extractor.cuda()
	cont_enc.cuda()
	stl_enc.cuda()
	dec.cuda()
	#decr.cuda()
	#mse_criterion.cuda()
	#classificationCriterion.cuda()
	#letterClassifierModel.cuda()


torch.manual_seed(0)



for i, data in enumerate(dataloader, 0):		

	if i%10 != 0:
		continue

	img1 = data['A']
	img2 = data['B']
	img12 = data['A2']
	img21 = data['B2']

	if opt.cuda:
		img1 = img1.cuda()
		img2 = img2.cuda()
		img12 = img12.cuda()
		img21 = img21.cuda()

	#path1 = data['A_paths'][0]
	#path2 = data['B_paths'][0]

	#bla(stl_enc, cont_enc, img1, )
	bla(stl_enc, cont_enc, img1, 1, 1, i)
	bla(stl_enc, cont_enc, img2, 2, 2, i)
	bla(stl_enc, cont_enc, img12, 1, 2, i)
	bla(stl_enc, cont_enc, img21, 2, 1, i)

	continue
	
	lst1 = stl_enc.encoder.children()
	x = img1
	for idx,l in enumerate(lst1):
		x = l(x)
		#print(x.size())
		if idx == 1:
			feats1 = x
		if idx == 3:
			feats3 = x
		if idx == 7:
			feats7 = x
		if idx == 10:
			feats10 = x

	spc1 = pca_feat_map(feats1[0])
	spc3 = pca_feat_map(feats3[0])
	spc7 = pca_feat_map(feats7[0])
	spc10 = pca_feat_map(feats10[0])


	lst2 = cont_enc.encoder.children()
	x = img1
	for idx,l in enumerate(lst2):
		x = l(x)
		#print(x.size())
		if idx == 1:
			feats1 = x
		if idx == 3:
			feats3 = x
		if idx == 7:
			feats7 = x
		if idx == 10:
			feats10 = x

	cpc1 = pca_feat_map(feats1[0])
	cpc3 = pca_feat_map(feats3[0])
	cpc7 = pca_feat_map(feats7[0])
	cpc10 = pca_feat_map(feats10[0])



	save_image(opt.out, i, 'stl', 1, spc1)
	save_image(opt.out, i, 'stl', 3, spc3)
	save_image(opt.out, i, 'stl', 7, spc7)
	save_image(opt.out, i, 'stl', 10, spc10)
	
	save_image(opt.out, i, 'cont', 1, cpc1)
	save_image(opt.out, i, 'cont', 3, cpc3)
	save_image(opt.out, i, 'cont', 7, cpc7)
	save_image(opt.out, i, 'cont', 10, cpc10)

	# imageio.imwrite(os.path.join(opt.out,'%d_stl_01.png' % (i)), spc1)
	# imageio.imwrite(os.path.join(opt.out,'%d_stl_03.png' % (i)), spc3)
	# imageio.imwrite(os.path.join(opt.out,'%d_stl_07.png' % (i)), spc7)
	# imageio.imwrite(os.path.join(opt.out,'%d_stl_10.png' % (i)), spc10)

	# imageio.imwrite(os.path.join(opt.out,'%d_cont_01.png' % (i)), cpc1)
	# imageio.imwrite(os.path.join(opt.out,'%d_cont_03.png' % (i)), cpc3)
	# imageio.imwrite(os.path.join(opt.out,'%d_cont_07.png' % (i)), cpc7)
	# imageio.imwrite(os.path.join(opt.out,'%d_cont_10.png' % (i)), cpc10)





