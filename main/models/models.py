""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .model_parts import *

# The human detector network
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.n_channel = 30
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_channel, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_channel, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_channel, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_channel, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_channel, out_channels=64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(
            input_size=128*1,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.linear = nn.Sequential(
            nn.Linear(128*5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x1 = self.conv1(x[:, 0:30, :, :])
        x2 = self.conv2(x[:, 30:60, :, :])
        x3 = self.conv3(x[:, 60:90, :, :])
        x4 = self.conv4(x[:, 90:120, :, :])
        x5 = self.conv5(x[:, 120:150, :, :])
        x1 = x1.reshape(x.size(0),-1)
        x2 = x2.reshape(x.size(0), -1)
        x3 = x3.reshape(x.size(0), -1)
        x4 = x4.reshape(x.size(0), -1)
        x5 = x5.reshape(x.size(0), -1)
        x = torch.stack([x1,x2,x3,x4,x5], dim=1)
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = F.dropout(x, p=0.5)
        x = torch.sigmoid(self.out(x))
        return x

# The human segmentor network
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input = DoubleConv(n_channels, 64, first_block=True)
        self.inc = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = DoubleConv(64, 32)
        self.output = OutConv(32, n_classes)

    def forward(self, x):
        x = self.input(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        output = self.output(x)
        return output

# The forgery attack detector network
class ResNet(nn.Module):
    def __init__(self, Residual, num_classes=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.layer1 = self.resnet_block(Residual, 32, 64, 2)
        self.layer2 = self.resnet_block(Residual, 64, 128, 2)
        self.layer3 = self.resnet_block(Residual, 128, 256, 2)
        self.layer4 = self.resnet_block(Residual, 256, 512, 2)

        #         self.res18 = nn.Sequential(self.conv1,
        #                                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        #                                    nn.BatchNorm2d(128),
        #                                    nn.ReLU(),
        #                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #                                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
        #                                    nn.BatchNorm2d(256),
        #                                    nn.ReLU(),
        #                                    nn.AdaptiveAvgPool2d((1,1)))

        self.res18 = nn.Sequential(
            self.conv1,
            self.layer1,
            self.layer2,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #             self.layer3,
            #             self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)))

        self.lstm = torch.nn.LSTM(
            input_size=128 * 1,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.linear = nn.Sequential(
            nn.Linear(128 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.fc = nn.Linear(128, num_classes)

    def resnet_block(self, Residual, in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, shortcut=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    def preprocess(self, x):
        # GOP = 3
        #         out_1 = torch.stack((x[:, 0, :, :], x[:, 3, :, :]), dim=1)
        #         out_2 = torch.stack((x[:, 1, :, :], x[:, 4, :, :]), dim=1)
        #         out_3 = torch.stack((x[:, 2, :, :], x[:, 5, :, :]), dim=1)
        # GOP = 5
        #         out_1 = torch.stack((x[:, 0, :, :], x[:, 5, :, :]), dim=1)
        #         out_2 = torch.stack((x[:, 1, :, :], x[:, 6, :, :]), dim=1)
        #         out_3 = torch.stack((x[:, 2, :, :], x[:, 7, :, :]), dim=1)
        #         out_4 = torch.stack((x[:, 3, :, :], x[:, 8, :, :]), dim=1)
        #         out_5 = torch.stack((x[:, 4, :, :], x[:, 9, :, :]), dim=1)
        # GOP = 7
        out_1 = torch.stack((x[:, 0, :, :], x[:, 7, :, :]), dim=1)
        out_2 = torch.stack((x[:, 1, :, :], x[:, 8, :, :]), dim=1)
        out_3 = torch.stack((x[:, 2, :, :], x[:, 9, :, :]), dim=1)
        out_4 = torch.stack((x[:, 3, :, :], x[:, 10, :, :]), dim=1)
        out_5 = torch.stack((x[:, 4, :, :], x[:, 11, :, :]), dim=1)
        out_6 = torch.stack((x[:, 5, :, :], x[:, 12, :, :]), dim=1)
        out_7 = torch.stack((x[:, 6, :, :], x[:, 13, :, :]), dim=1)

        return out_1, out_2, out_3, out_4, out_5, out_6, out_7

    def forward(self, x):
        out_1, out_2, out_3, out_4, out_5, out_6, out_7 = self.preprocess(x)
        out_1 = self.res18(out_1)
        with torch.no_grad():
            self.res18.eval()
            out_2 = self.res18(out_2)
            out_3 = self.res18(out_3)
            out_4 = self.res18(out_4)
            out_5 = self.res18(out_5)
            out_6 = self.res18(out_6)
            out_7 = self.res18(out_7)
            self.res18.train()
        out_1 = out_1.reshape(out_1.size(0), -1)
        out_2 = out_2.reshape(out_2.size(0), -1)
        out_3 = out_3.reshape(out_3.size(0), -1)
        out_4 = out_4.reshape(out_4.size(0), -1)
        out_5 = out_5.reshape(out_5.size(0), -1)
        out_6 = out_6.reshape(out_6.size(0), -1)
        out_7 = out_7.reshape(out_7.size(0), -1)
        out = torch.stack([out_1, out_2, out_3, out_4, out_5, out_6, out_7], dim=1)
        out, _ = self.lstm(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        out = self.fc(out)
        return out





