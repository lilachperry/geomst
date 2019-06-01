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


parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
#parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=16, help='the low resolution image size')
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
opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTest/may7LastNailInCoffin_Epoch70'


if not os.path.exists(opt.out):
    os.makedirs(opt.out)

opt.hasGroundTruth = False
opt.invertColorsContent = False
opt.phase = 'test'
opt.dataset_mode = 'unaligned'
#opt.indir = '/mnt/data/messin/dev/python/autoEncoder/outTrain/may3_TrainPatchesBreakable_imsize16/checkpoints'
#opt.indir = '/mnt/data/messin/dev/python/autoEncoder/outTrain/may6_TrainPatchesBreakable_imsize16/checkpoints'
opt.indir = '/mnt/data/messin/dev/python/autoEncoder/outTrain/may6_imsize16_Akronim/checkpoints'
#opt.dataroot = '/mnt/data/noafish/geomst/datasets/fonts'
#opt.dataroot = '/home/messin/dev/python/geomst/dataset/clusterFeatures/letterClassifierLayerPar9_apr24_2/cluster9/Images'
opt.datarootContent = '/mnt/data/messin/dev/matlab/sketchesContextTest2/noContext/16'
opt.datarootStyle = '/home/messin/dev/python/geomst/dataset/pairs/Akronim/train'
#opt.datarootContent = '/home/messin/dev/python/geomst/dataset/Metropolis-SemiBoldItalic'
#opt.datarootStyle = '/home/messin/dev/python/geomst/dataset/clusterFeatures/letterClassifierLayerPar9_apr25/cluster3/chosen/'
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
opt.use_concat = True
cont_enc_path = os.path.join(opt.indir, 'cont_enc_latest.pth')
stl_enc_path = os.path.join(opt.indir, 'stl_enc_latest.pth')
dec_path = os.path.join(opt.indir, 'dec_latest.pth')

#lfile = os.path.join(opt.out, 'loss_log.txt')
#lfile_handle = open(lfile, 'w')


cont_enc = _EncoderNoa(opt.imageSize)
print(cont_enc)
stl_enc = _EncoderNoa(opt.imageSize)
dec = _DecoderNoa(opt.imageSize, opt.use_concat)
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


dataCnt = 0
for i, data in enumerate(dataloader, 0):

    # contentIm = data['content']
    # styleIm = data['style']

    contentIm = data['style']
    styleIm = data['content']

    if opt.cuda:
        contentIm = contentIm.cuda()
        styleIm = styleIm.cuda()

    stl = stl_enc(styleIm)
    cont = cont_enc(contentIm)

    stlcont = torch.cat((stl, cont), 1)

    decIm = dec(stlcont)

    for j in range(len(decIm)):
        im1 = util.tensor2im(decIm[j])
        origCont = util.tensor2im(contentIm[j])
        origStyle = util.tensor2im(styleIm[j])

        imageio.imwrite(os.path.join(opt.out,'%03dFake.png' % (dataCnt)), im1)
        imageio.imwrite(os.path.join(opt.out,'%03dOrigContent.png' % (dataCnt)), origCont)
        imageio.imwrite(os.path.join(opt.out,'%03dOrigStyle.png' % (dataCnt)), origStyle)
        dataCnt = dataCnt + 1


# if epoch%10 == 0:
# 	torch.save(cont_enc.state_dict(), '%s/cont_enc_epoch_%d.pth' % (opt.out, epoch))
# 	torch.save(stl_enc.state_dict(), '%s/stl_enc_epoch_%d.pth' % (opt.out, epoch))
# 	torch.save(dec.state_dict(), '%s/dec_epoch_%d.pth' % (opt.out, epoch))

