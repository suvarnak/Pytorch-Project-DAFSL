"""
DCGAN discriminator model
based on the paper: https://arxiv.org/pdf/1511.06434.pdf
date: 30 April 2018
"""
import torch
import torch.nn as nn
import torchvision.models as models
import json
from easydict import EasyDict as edict
from graphs.weights_initializer import weights_init


class DAFSL_ConceptDiscriminatorModel(nn.Module):
    def __init__(self, config,no_class_labels):
        super().__init__()
        self.model_ft = models.vgg11_bn(pretrained=True)
        lt=8
        cntr=0
        for child in self.model_ft.children():
            cntr+=1
            if cntr < lt:
						# print child
                for param in child.parameters():
                    param.requires_grad = False

        num_ftrs = self.model_ft.classifier[6].in_features
        self.model_ft.classifier[6] = nn.Linear(num_ftrs,no_class_labels)

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1, padding=1),  # b, 32, 224, 224
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 32, 112, 112
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1),  # b, 64, 112, 112
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None),  # b, 64, 56, 56
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # b, 128, 56, 56
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=None)  # b, 128, 28, 28
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, no_class_labels) #self.config.no_of_classes)
        )


    def forward(self, x):
        x = self.model_ft(x)
        #x = self.cnn_layers(x)
        #x = x.view(x.size(0), -1)
        #x = self.linear_layers(x)
        return x

"""
netD testing
"""
def main():
    config = json.load(open('../../configs/dafsl_exp_0.json'))
    config = edict(config)
    inp  = torch.autograd.Variable(torch.randn(config.batch_size, config.input_channels, config.image_size, config.image_size))
    print (inp.shape)
    netD = DAFSL_ConceptDiscriminatorModel(config)
    out = netD(inp)
    print (out)

if __name__ == '__main__':
    main()
