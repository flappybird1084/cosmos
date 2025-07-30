from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd# import everything ml
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import re
import string
from wordcloud import WordCloud
import nltk
from collections import Counter
import contractions
from tqdm import tqdm
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from PIL import Image, ImageFile
import torchvision.models as models
from PIL import ImageDraw
import datasets
from datasets import load_dataset
from torch.utils.data import random_split



class DoubleConv(nn.Module):
    def __init__(self, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], use_sigmoid=True):
        self.use_sigmoid = use_sigmoid
        
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature))  # after concatenation

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid() if out_channels == 1 else nn.Identity()
    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # upsample
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip_connection, x), dim=1)  # concat
            x = self.decoder[idx + 1](x)  # double conv

        # return self.final_conv(x)
        if(self.use_sigmoid):
            return self.output_activation(self.final_conv(x))
        else:
            return self.final_conv(x)

            
# Define transform
transform_pipeline = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu") # sue me

heightmap_gen_model = UNet(in_channels=3, out_channels=1, use_sigmoid=False, features=[64,128,256,512, 1024]).to(device)
terrain_gen_model = UNet(in_channels=3, out_channels=3).to(device)

heightmap_gen_model.load_state_dict(torch.load('./models/terrain/turbo_heightmap_unet_model.pth', map_location=device))
terrain_gen_model.load_state_dict(torch.load('./models/terrain/turbo_terrain_unet_model.pth', map_location=device))

