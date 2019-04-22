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
from models import _EncoderNoa, _DecoderNoa, FontClasifier_1_4, KfirFeatureExtractor
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



def my_mse(a, b):
    d = a - b
    d = torch.mean(d ** 2)
    return d

def myL1(a,b):
    return torch.mean(torch.abs(a - b))

def writeModel(f, model, modelName):
    f.write('%s:\n'%modelName)
    f.write(str(model))
    f.write('\n\n======================================================================================================\n\n')

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
#parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--nClasses', type=int, default=132, help='number of classes')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the low resolution image size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--MSEWeight', type=int, default=100, help='the weight of the MSE loss')
parser.add_argument('--learningRate', type=float, default=0.001, help='learning rate for optimizer')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
parser.add_argument('--tripletLossMargin', type=float, default=1, help='margin for triplet loss')
parser.add_argument('--FMLayerParameter', type=int, default=10, help='the layer in the classifier from which the features will be extracted')



opt = parser.parse_args()
print(opt)

useConcat = True
classifyFonts = True
useFeatureMatchingLoss = True
useMSE = False
mar = 1


if classifyFonts:
    nClasses = opt.nClasses
else:
    nClasses = 26
#useLetterClassificationLoss = False

#modelPath = '/home/messin/dev/python/Pytorch-srgan/fontClassifier/trainedModels/trainFontClassifier_may2_uniqueSubset132_b32_class1_4/checkpoints/classifier_ep199.pth'
#letterClassificationModelPath = '/mnt/data/messin/dev/python/letterClassifier/outTrain/Feb4_b32_26classFix/checkpoints/classifier_ep23.pth'
fontClassifierModelPath = '/home/messin/dev/python/Pytorch-srgan/fontClassifier/trainedModels/trainFontClassifier_may2_uniqueSubset132_b32_class1_4/checkpoints/classifier_ep199.pth'

#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/train_spirax_batch#16_imSize#28_Jan20_RefWeight50To200/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/trainfullLetter_Jan29_batch32_imSize28_Normalized_sigmoidInsteadOfTanh_MSEandFM/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/toDelete/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/trainCross_Apr2_normFix_cycleLoss/'
#opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr15_L1withFMLossMargin1_lastConvInsteadOfLastRelu/'
opt.out = '/mnt/data/messin/dev/python/autoEncoder/outTrain/apr22_L1Margin1_FMPar10/'

checkpointsDir = '%s/checkpoints/' % opt.out
resultsDir = '%s/results/' % opt.out

folders = [opt.out, checkpointsDir, resultsDir]
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)


opt.phase = 'train'
opt.dataset_mode = 'unaligned'
#opt.dataroot = '/mnt/data/noafish/geomst/datasets/fonts'
opt.dataroot = '/home/messin/dev/python/geomst/dataset/fonts'
opt.resize_or_crop = 'resize_and_crop'
opt.fineSize = opt.imageSize #64
opt.loadSize = opt.imageSize #64
opt.isTrain = True
opt.no_flip = True
opt.gray = False
opt.serial_batches = False
opt.nThreads = 4
opt.max_dataset_size = float("inf")
#opt.which_direction = 'AToB'
opt.input_nc = 1
opt.nc = 1
opt.useConcat = useConcat
opt.classifyFonts = classifyFonts
opt.useFeatureMatchingLoss = useFeatureMatchingLoss
opt.useMSE = useMSE
data_loader = CreateDataLoader(opt)
dataloader = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)



lfile = os.path.join(opt.out, 'loss_log.txt')
lfile_handle = open(lfile, 'w')

optFile = os.path.join(opt.out, 'opt.txt')
optFile_handle = open(optFile, 'w')
optFile_handle.write(str(opt))

modelFile = os.path.join(opt.out, 'models.txt')
modelFile_handle = open(modelFile, 'w')


with torch.no_grad():
    # classifier = FontClasifier_1_4(nClasses, classifyFonts)
    # if opt.cuda:
    #     classifier.cuda()
    # classifier.load_state_dict(torch.load(fontClassifierModelPath))
    # classifier.eval()

    # For the feature matching loss
    classifier = FontClasifier_1_4(nClasses, classifyFonts)
    classifier.load_state_dict(torch.load(fontClassifierModelPath))
    feature_extractor = KfirFeatureExtractor(classifier, opt)
    print(feature_extractor)


cont_enc = _EncoderNoa(opt.imageSize)
stl_enc = _EncoderNoa(opt.imageSize)
dec = _DecoderNoa(opt.imageSize, useConcat)

writeModel(modelFile_handle, feature_extractor, "feature_extractor")
writeModel(modelFile_handle, classifier, "classifier")
writeModel(modelFile_handle, cont_enc, "cont_enc")
writeModel(modelFile_handle, stl_enc, "stl_enc")
writeModel(modelFile_handle, dec, "dec")


#decr = _DecoderNoa(opt.imageSize)
mse_criterion = nn.MSELoss()
L1criterion = nn.L1Loss()
class_criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

optimizer_cont_enc = torch.optim.Adam(cont_enc.parameters(), lr=opt.learningRate, weight_decay=1e-5)
optimizer_stl_enc = torch.optim.Adam(stl_enc.parameters(), lr=opt.learningRate, weight_decay=1e-5)
optimizer_dec = torch.optim.Adam(dec.parameters(), lr=opt.learningRate, weight_decay=1e-5)
#optimizer_decr = torch.optim.Adam(decr.parameters(), lr=opt.learningRate, weight_decay=1e-5)


if opt.cuda:
    feature_extractor.cuda()
    cont_enc.cuda()
    stl_enc.cuda()
    dec.cuda()
    #decr.cuda()
    mse_criterion.cuda()
    L1criterion.cuda()
#classificationCriterion.cuda()
#letterClassifierModel.cuda()


torch.manual_seed(0)


for epoch in range(opt.nEpochs):
    for i, data in enumerate(dataloader, 0):

        img1 = data['A']
        img2 = data['B']
        img12 = data['A2'] #style1Content2
        img21 = data['B2'] #style2Content1

        if opt.cuda:
            img1 = img1.cuda()
            img2 = img2.cuda()
            img12 = img12.cuda() #style1Content2
            img21 = img21.cuda() #style2Content1

        #path1 = data['A_paths'][0]
        #path2 = data['B_paths'][0]

        stl1 = stl_enc(img1)
        stl2 = stl_enc(img2)
        cont1 = cont_enc(img1)
        cont2 = cont_enc(img2)

        stl12 = stl_enc(img12)
        stl21 = stl_enc(img21)
        cont12 = cont_enc(img12) #style1Content2
        cont21 = cont_enc(img21) #style2Content1

        if useConcat:
            stl1cont2 = torch.cat((stl1, cont2), 1)
            stl2cont1 = torch.cat((stl2, cont1), 1)
            stl1cont1 = torch.cat((stl1, cont1), 1)
            stl2cont2 = torch.cat((stl2, cont2), 1)
        else:
            stl1cont2 = stl1 + cont2
            stl2cont1 = stl2 + cont1
            stl1cont1 = stl1 + cont1
            stl2cont2 = stl2 + cont2

        dec12 = dec(stl1cont2)
        dec21 = dec(stl2cont1)
        dec11 = dec(stl1cont1)
        dec22 = dec(stl2cont2)

        if useMSE:
            loss12 = mse_criterion(dec12, img12) #style1Content2
            loss21 = mse_criterion(dec21, img21) #style2Content1
            loss11 = mse_criterion(dec11, img1) #stlye1Content1
            loss22 = mse_criterion(dec22, img2)#stlye2Content2
        else:
            loss12 = L1criterion(dec12, img12)  # style1Content2
            loss21 = L1criterion(dec21, img21)  # style2Content1
            loss11 = L1criterion(dec11, img1)  # stlye1Content1
            loss22 = L1criterion(dec22, img2)  # stlye2Content2


        # stl1cont2r = torch.cat((cont2, stl1), 1)
        # stl2cont1r = torch.cat((cont1, stl2), 1)
        # stl1cont1r = torch.cat((cont1, stl1), 1)
        # stl2cont2r = torch.cat((cont2, stl2), 1)

        # dec12r = decr(stl1cont2r)
        # dec21r = decr(stl2cont1r)
        # dec11r = decr(stl1cont1r)
        # dec22r = decr(stl2cont2r)

        # loss12r = mse_criterion(dec12r, img12)
        # loss21r = mse_criterion(dec21r, img21)
        # loss11r = mse_criterion(dec11r, img1)
        # loss22r = mse_criterion(dec22r, img2)


        d1 = my_mse(stl1, stl12) #same style - style 1 different content
        d2 = my_mse(stl1, stl21) #different style same content
        loss1 = torch.clamp(d1 - d2 + mar, min=0.0)
        d3 = my_mse(stl2, stl21) #same style - style2 diffferent content
        d4 = my_mse(stl2, stl12)  #different style same content
        loss2 = torch.clamp(d3 - d4 + mar, min=0.0)
        d5 = my_mse(cont1, cont21) #same content  - content 1 different style
        d6 = my_mse(cont2, cont21) #different content same style
        loss3 = torch.clamp(d5 - d6 + mar, min=0.0)
        d7 = my_mse(cont2, cont12) #same content  - content2 different style
        d8 = my_mse(cont1, cont12) #different content same styleFMLayerParameter
        loss4 = torch.clamp(d7 - d8 + mar, min=0.0)

        trip_loss = loss1 + loss2 + loss3 + loss4

        if useFeatureMatchingLoss:
            level = opt.FMLayerParameter
            style1Content2FeaturesOrig = feature_extractor(img12, level=level, start_level=0)
            style2Content1FeaturesOrig = feature_extractor(img21, level=level, start_level=0)
            style1Content2FeaturesOutput = feature_extractor(dec12, level=level, start_level=0)
            style2Content1FeaturesOutput = feature_extractor(dec21, level=level, start_level=0)
            # fmLossStyle1Content2 = 0.00000001 * mse_criterion(style1Content2FeaturesOutput, style1Content2FeaturesOrig)
            # fmLossStyle2Content1 = 0.00000001 * mse_criterion(style2Content1FeaturesOutput, style2Content1FeaturesOrig)
            fmLossStyle1Content2 = 0.000001*mse_criterion(style1Content2FeaturesOutput, style1Content2FeaturesOrig)
            fmLossStyle2Content1 = 0.000001*mse_criterion(style2Content1FeaturesOutput, style2Content1FeaturesOrig)
        else:
            fmLossStyle1Content2 = 0
            fmLossStyle2Content1 = 0

        #loss = loss12 + loss21 + loss11 + loss22 + loss12r + loss21r + loss11r + loss22r + trip_loss
        loss = loss12 + loss21 + loss11 + loss22 + trip_loss + fmLossStyle1Content2 + fmLossStyle2Content1


        optimizer_dec.zero_grad()
        #optimizer_decr.zero_grad()
        optimizer_stl_enc.zero_grad()
        optimizer_cont_enc.zero_grad()
        loss.backward()
        optimizer_dec.step()
        #optimizer_decr.step()
        optimizer_stl_enc.step()
        optimizer_cont_enc.step()


        if i % 1000 == 0:
            im1 = util.tensor2im(img1[0])
            im2 = util.tensor2im(img2[0])
            oim12 = util.tensor2im(img12[0])
            oim21 = util.tensor2im(img21[0])
            im12 = util.tensor2im(dec12[0])
            im21 = util.tensor2im(dec21[0])

            imageio.imwrite(os.path.join(resultsDir,'%d_%d_style_1_cont_2.png' % (epoch, i)), im12)
            imageio.imwrite(os.path.join(resultsDir,'%d_%d_style_2_cont_1.png' % (epoch, i)), im21)
            imageio.imwrite(os.path.join(resultsDir,'%d_%d_style_1_cont_1_orig.png' % (epoch, i)), im1)
            imageio.imwrite(os.path.join(resultsDir,'%d_%d_style_2_cont_2_orig.png' % (epoch, i)), im2)
            imageio.imwrite(os.path.join(resultsDir,'%d_%d_style_1_cont_2_orig.png' % (epoch, i)), oim12)
            imageio.imwrite(os.path.join(resultsDir,'%d_%d_style_2_cont_1_orig.png' % (epoch, i)), oim21)

            if useFeatureMatchingLoss:
                _, predicted = torch.max(style1Content2FeaturesOrig.data, 1)
                _, predicted2 = torch.max(style1Content2FeaturesOutput.data, 1)
                print(predicted.data)
                print('/n')
                print(predicted2.data)
                print('/n')


        #loss_str = '[%d/%d][%d/%d] Loss_12: %.4f Loss_21: %.4f Loss_11: %.4f Loss_22: %.4f \n' % (epoch, opt.nEpochs, i, len(dataloader), loss12, loss21, loss11, loss22)
        loss_str = '[%d/%d][%d/%d] Loss_12: %.4f Loss_21: %.4f Loss_11: %.4f Loss_22: %.4f Loss_t1: %.4f Loss_t2: %.4f Loss_c1: %.4f Loss_c2: %.4f Loss_fm1: %.4f Loss_fm2: %.4f\n' % (epoch, opt.nEpochs, i, len(dataloader), loss12, loss21, loss11, loss22, loss1, loss2, loss3, loss4, fmLossStyle1Content2, fmLossStyle2Content1)
        print(loss_str)
        lfile_handle.write(loss_str)
    #print('[%d/%d][%d/%d] Loss_12: %.4f Loss_21: %.4f Loss_11: %.4f Loss_22: %.4f)' % (epoch, opt.nEpochs, i, len(dataloader), loss12, loss21, loss11, loss22))


    torch.save(cont_enc.state_dict(), '%s/cont_enc_latest.pth' % (checkpointsDir))
    torch.save(stl_enc.state_dict(), '%s/stl_enc_latest.pth' % (checkpointsDir))
    torch.save(dec.state_dict(), '%s/dec_latest.pth' % (checkpointsDir))


    if epoch%10 == 0:
        torch.save(cont_enc.state_dict(), '%s/cont_enc_epoch_%d.pth' % (checkpointsDir, epoch))
        torch.save(stl_enc.state_dict(), '%s/stl_enc_epoch_%d.pth' % (checkpointsDir, epoch))
        torch.save(dec.state_dict(), '%s/dec_epoch_%d.pth' % (checkpointsDir, epoch))

