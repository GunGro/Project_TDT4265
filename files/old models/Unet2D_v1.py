import torch
from torch import nn
from torchvision import models, datasets, transforms

class Unet2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.resnet = models.resnet34(pretrained=True)
        n_image_channels = in_channels
        self.resnet.conv1 = nn.Conv2d(n_image_channels,
                                self.resnet.conv1.out_channels,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False)

        self.conv1 = self.contract_block(256, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)


#3 torch.Size([1, 64, 96, 96])
#4 torch.Size([1, 64, 96, 96])
#5 torch.Size([1, 128, 48, 48])
#6 torch.Size([1, 256, 24, 24])


        self.upconv3 = self.expand_block(128, 128, 3, 1)
        self.upconv2 = self.expand_block(64+128, 64, 3, 1)
        self.upconv1 = self.expand_block(32+64, 128, 3, 1)
        self.upconv0 = self.expand_block(128, 64, 3, 1)


        self.upconvRes1 = self.expand_block(256+128, 256, 3, 1)
        self.upconvRes2 = self.expand_block(128+256, 512, 3, 1)
        self.upconvRes3 = self.expand_block(64+512, 512, 3, 1)
        self.upconvRes4 = self.expand_block(64+512, out_channels, 3, 1)



    def __call__(self, x):
        # pretrained feature detector
        for i, child in enumerate(self.resnet.children()):
            x = child.forward(x)

            if (i==1):
                out1 = x
            if (i==3):
                out3 = x
            if (i==4):
                out4 = x
            if (i==5):
                out5 = x
            if (i==6):
                out6 = x

            if (i==6):
                break
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        #upsample
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        upconvRes1 = self.upconvRes1(torch.cat([upconv1, out6], 1))
        upconvRes2 = self.upconvRes2(torch.cat([upconvRes1, out5], 1))
        upconvRes3 = self.upconvRes3(torch.cat([upconvRes2, out4], 1))
        upconvRes4 = self.upconvRes4(torch.cat([upconvRes3, out1], 1))

        return upconvRes4

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),

                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand


if __name__ == "__main__":

    # define model

    unet = Unet2D(1,4)

    # print

    #print(unet)

    resnet = next(unet.children())
    for i, (name, child) in enumerate(resnet.named_children()):

        print(i, name, child)

