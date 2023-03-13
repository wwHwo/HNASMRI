import torch.nn as nn
import torch
from model.dagan import UNet
import os
from torchsummary import summary
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modle = UNet().to(device)
summary(modle, input_size=(1, 256, 256))
