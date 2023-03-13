import torch
import torch.nn as nn
from torch.nn import Module


class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.dif_dim = 64
        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.dif_dim, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim, out_channels=self.dif_dim * 2, kernel_size=(4, 4), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 2, out_channels=self.dif_dim * 4, kernel_size=(4, 4), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 4, out_channels=self.dif_dim * 8, kernel_size=(4, 4), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 8, out_channels=self.dif_dim * 16, kernel_size=(4, 4), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 16, out_channels=self.dif_dim * 32, kernel_size=(4, 4), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 32),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 32, out_channels=self.dif_dim * 16, kernel_size=(1, 1), stride=(1, 1),
                      padding=0),
            nn.BatchNorm2d(num_features=self.dif_dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Conv7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 16, out_channels=self.dif_dim * 8, kernel_size=(1, 1), stride=(1, 1),
                      padding=0),
            nn.BatchNorm2d(num_features=self.dif_dim * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.Res8 = nn.Sequential(
            nn.Conv2d(in_channels=self.dif_dim * 8, out_channels=self.dif_dim * 2, kernel_size=(1, 1), stride=(1, 1),
                      padding=0),
            nn.BatchNorm2d(num_features=self.dif_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.dif_dim * 2, out_channels=self.dif_dim * 2, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=self.dif_dim * 2, out_channels=self.dif_dim * 8, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(num_features=self.dif_dim * 8),
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.Classifier = nn.Sequential(
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        h0 = self.Conv0(input)
        h1 = self.Conv1(h0)
        h2 = self.Conv2(h1)
        h3 = self.Conv3(h2)
        h4 = self.Conv4(h3)
        h5 = self.Conv5(h4)
        h6 = self.Conv6(h5)
        h7 = self.Conv7(h6)
        r8 = self.Res8(h7)
        h8 = torch.add(input=h7, other=r8)
        h8 = self.relu(h8)
        h9 = h8.contiguous().view(h8.size(0), -1)
        output = self.Classifier(h9)
        return output


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 128*128*64
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 64*64*128
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 32*32*256
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 16*16*512
        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 8*8*512
        self.Conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 4*4*512
        self.Conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 2*2*512
        self.Conv8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 1*1*512
        self.DeConv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )  # 2*2*512
        self.DeConv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU()
        )  # 4*4*1024
        self.DeConv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1536, out_channels=1024, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU()
        )  # 8*8*1024
        self.DeConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1536, out_channels=1024, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU()
        )  # 16*16*1024
        self.DeConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1536, out_channels=256, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )  # 32*32*256
        self.DeConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )  # 64*64*128
        self.DeConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )  # 128*128*64
        self.DeConv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, padding=1, kernel_size=4,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )  # 256*256*64
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )
        self.Tanh = nn.Tanh()

    def forward(self, input):
        down1 = self.Conv1(input)
        down2 = self.Conv2(down1)
        down3 = self.Conv3(down2)
        down4 = self.Conv4(down3)
        down5 = self.Conv5(down4)
        down6 = self.Conv6(down5)
        down7 = self.Conv7(down6)
        down8 = self.Conv8(down7)
        up7 = self.DeConv7(down8)
        up6 = self.DeConv6(torch.cat((down7, up7), 1))
        up5 = self.DeConv5(torch.cat((down6, up6), 1))
        up4 = self.DeConv4(torch.cat((down5, up5), 1))
        up3 = self.DeConv3(torch.cat((down4, up4), 1))
        up2 = self.DeConv2(torch.cat((down3, up3), 1))
        up1 = self.DeConv1(torch.cat((down2, up2), 1))
        up0 = self.DeConv0(torch.cat((down1, up1), 1))
        output = self.out(up0)
        output = torch.add(output, input)
        output = self.Tanh(output)
        return output
