import pretrainedmodels         # using pretrained model
import torch.nn as nn
from torch.nn import functional as F

class ResNet34(nn.Module):      # model defined in pytorch using nn.Module
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained="imagenet")    # base model
        else:
                        self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)    # base model

        self.l0 = nn.Linear(512, 168)   # final layer for grapheme, with 512 inputs and 168 outputs (grapheme classes)
        self.l1 = nn.Linear(512, 11)    # final layer for vowel, with 512 inputs and 11 outputs (vowel classes)
        self.l2 = nn.Linear(512, 7)     # final layer for consonants, with 512 inputs and 7 outputs (consonant classes)

    def forward(self, x):           # this takes the batch x
        bs, _, _, _ = x.shape       # need: batchsize | dont_need: channels, height and width
        x = self.model.features(x)  # extract features using the feature function
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)    # adaptive average pooling
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2
    """ we remove the final layer of the pretrained model and add a linear layer for the 3 classes.
    This is done so that the pretrained models will be able to predict the three classes"""