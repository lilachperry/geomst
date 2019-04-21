import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        #b,32,1,64,64
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  # b, 32, 32,32
        self.conv2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)  # b, 16, 8, 8
        self.conv3 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 4, 4
        self.maxpool1 = nn.MaxPool2d(2, stride=2)  # b, 32, 16, 16
        self.maxpool2 = nn.MaxPool2d(2, stride=1)  # b, 16, 7, 7
        self.maxpool3 = nn.MaxPool2d(2, stride=1)  # b, 8, 3, 3

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        return x


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.convTrans1 = nn.ConvTranspose2d(8, 32, 4, stride=2)  # b, 32, 8, 8
        self.convTrans2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # b, 16, 16, 16
        self.convTrans3 = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)  # b, 8, 32, 32
        self.convTrans4 = nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)  # b, 1, 64, 64

    def forward(self, x):
        x = F.relu(self.convTrans1(x)) # b, 32, 8, 8
        x = F.relu(self.convTrans2(x)) # b, 16, 16, 16
        x = F.tanh(self.convTrans3(x)) # b, 8, 32, 32
        x = F.tanh(self.convTrans4(x)) #
        #x = F.sigmoid(self.convTrans3(x))

        return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = encoder()
        self.decoder = decoder()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)

        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x


class FontClasifier_1_4(nn.Module):

    def __init__(self, nClasses, classifyFonts):
        super(FontClasifier_1_4, self).__init__()
        # conv2d(in_channels, outChannels, kernel_size
        self.conv1 = nn.Conv2d(1, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (2, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (1, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32, 48, 2)
        self.bn4 = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 48, 2)
        self.bn5 = nn.BatchNorm2d(48)
        self.conv6 = nn.Conv2d(48, 48, 2)
        self.bn6 = nn.BatchNorm2d(48)

        self.conv7 = nn.Conv2d(48, 80, 2)
        self.bn7 = nn.BatchNorm2d(80)
        self.conv8 = nn.Conv2d(80, 80, 2)
        self.bn8 = nn.BatchNorm2d(80)
        self.conv9 = nn.Conv2d(80, 80, 2)
        self.bn9 = nn.BatchNorm2d(80)

        self.fc1 = nn.Linear(80 * 5 * 5, 800)
        if classifyFonts:
            self.fc2 = nn.Linear(800, 500)
            self.fc3 = nn.Linear(500, nClasses)
        else:
            self.fc2 = nn.Linear(800, 300)
            self.fc3 = nn.Linear(300, nClasses)

    #@profile
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(F.relu(self.bn9(self.conv9(x))))

        x = x.view(-1, 80 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def define_kfirFeatureExtractor(old_model, opt):
    #vgg19net = None
    use_gpu = len(opt.gpu_ids) > 0

    featsnet = KfirFeatureExtractor(old_model, opt)

    if use_gpu:
        assert(torch.cuda.is_available())
        featsnet.cuda(device_id=opt.gpu_ids[0])
    return featsnet

class KfirFeatureExtractor(nn.Module):
    def __init__(self, old_model,opt):
        super(KfirFeatureExtractor, self).__init__()
        #BaseModel.initialize(self, opt)


        layers = []
        layers = [old_model.conv1, old_model.bn1, nn.ReLU()]
        self.layer_0 = nn.Sequential(*layers)

        layers = [old_model.conv2, old_model.bn2, nn.ReLU()]
        self.layer_1 = nn.Sequential(*layers)

        layers = [old_model.conv3, old_model.bn3, nn.ReLU()]
        self.layer_2 = nn.Sequential(*layers)

        layers = [old_model.pool, old_model.conv4, old_model.bn4, nn.ReLU()]
        self.layer_3 = nn.Sequential(*layers)

        layers = [old_model.conv5, old_model.bn5, nn.ReLU()]
        self.layer_4 = nn.Sequential(*layers)

        layers = [old_model.conv6, old_model.bn6, nn.ReLU()]
        self.layer_5 = nn.Sequential(*layers)

        layers = [old_model.pool, old_model.conv7, old_model.bn7, nn.ReLU()]
        self.layer_6 = nn.Sequential(*layers)

        layers = [old_model.conv8, old_model.bn8, nn.ReLU()]
        self.layer_7 = nn.Sequential(*layers)

        layers = [old_model.conv9]
        self.layer_8 = nn.Sequential(*layers)

        layers = [old_model.bn9, nn.ReLU()]
        self.layer_9 = nn.Sequential(*layers)

        self.layers = [ self.layer_0, self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5, self.layer_6, self.layer_7,
                       self.layer_8, self.layer_9 ]

        # self.layer_1 = nn.Sequential()#self.make_layers(old_model,0,3)
        # self.layer_2 = self.make_layers(old_model,3,6)
        # self.layer_3 = self.make_layers(old_model,6,9)
        # self.layer_4 = self.make_layers(old_model,9,13)
        # self.layer_5 = self.make_layers(old_model, 13, 16)
        # self.layer_6 = self.make_layers(old_model, 16, 19)
        # self.layer_7 = self.make_layers(old_model, 19, 23)
        # self.layer_8 = self.make_layers(old_model, 23, 26)
        # self.layer_9 = self.make_layers(old_model, 26, 29)
        #
        # self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5, self.layer_6, self.layer_7, self.layer_8, self.layer_9]

        # self.run_cuda = 0
        # if opt.gpu_ids:
        #     self.run_cuda = 1

        #image_height = image_width = opt.fineSize
        #self.input = self.Tensor(opt.batchSize, opt.input_nc, image_height, image_width)
        #self.input = torch.cuda.FloatTensor(1,1,64,64,64) if self.run_cuda else torch.Tensor(1, 1, 64, 64, 64)
        #self.convergence_threshold = opt.convergence_threshold
        #self.old_lr = opt.lr
        #self.beta = opt.beta1

    # def make_layers(self, old_model, start_layer, end_layer):
    #     layer = []
    #     features = next(old_model.children())
    #     original_layer_number = 0
    #     for module in features.children():
    #         if original_layer_number >= start_layer and original_layer_number < end_layer:
    #             layer += [module]
    #         original_layer_number += 1
    #     return nn.Sequential(*layer)


    #def forward(self, level=5, start_level=0, set_as_var=True):
    #@profile
    def forward(self, input_sample, level=10, start_level=0):
        #assert (level >= start_level)
        # if set_as_var == True:
        #     self.input_sample = Variable(self.input)
        # else:
        #     self.input_sample = self.input

        #layer_i_output = layer_i_input = self.input_sample
        layer_i_output = layer_i_input = input_sample
        for i in range(start_level, level):
            layer_i = self.layers[i]
            layer_i_output = layer_i(layer_i_input)
            layer_i_input = layer_i_output

        return layer_i_output



nz = 100
ngf = 64
ndf = 64
nc = 1

class _EncoderNoa(nn.Module):
    def __init__(self, imageSize):
        super(_EncoderNoa, self).__init__()

        n = math.log(imageSize, 2)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        # self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        # self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        
        self.conv1 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(n - 3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                    nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            self.encoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))


        # self.layers = []
        # self.layers.append(nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        # self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        # for i in range(n - 3):
        #     # state size. (ngf) x 32 x 32
        #     self.layers.append(nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
        #     self.layers.append(nn.BatchNorm2d(ngf * 2 ** (i + 1)))
        #     self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        # self.layers.append(nn.Conv2d(ngf * 2 ** (n - 3), nz, 4))
        # self.layer_module = nn.ModuleList(self.layers)

        # state size. (ngf*8) x 4 x 4

    def forward(self, input):
        output = self.encoder(input)
        # print("output sz: ", output.size())
        # o1 = self.conv1(output)
        # o2 = self.conv2(output)
        # print("o1 sz: ", o1.size())
        # print("o2 sz: ", o2.size())
        # return [self.conv1(output),self.conv2(output)]
        # return [o1, o2]
        return self.conv1(output)


    def forward2(self, input):
        out = input
        for layer in self.layer_module:
            out = layer(out)

        return out


    def forward_layers(self, x):
        list = []
        for idx, name, module in enumerate(self._modules.iteritems()):
            x = module(x)
            print(idx)
            #if name in self._to_select:
            #    list.append(x)
        #return list
        return x


class _DecoderNoa(nn.Module):
    def __init__(self, imageSize, useConcat):
        super(_DecoderNoa, self).__init__()

        n = math.log(imageSize, 2)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        if useConcat:
            self.decoder.add_module('input-conv', nn.ConvTranspose2d(2*nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=False))
        else:
            self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (n - 3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n - 3, 0, -1):
            self.decoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i - 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i - 1)))
            self.decoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i - 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())


    def forward(self, input):
       return self.decoder(input)





class autoencoderNoa(nn.Module):
    def __init__(self, imageSize):
        super(autoencoderNoa, self).__init__()

        self.encoder = _EncoderNoa(imageSize)
        self.decoder = _DecoderNoa(imageSize)



    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)

        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x


