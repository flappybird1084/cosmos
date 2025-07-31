from IPython.display import display # pyright: ignore[reportMissingModuleSource, reportMissingImports] 
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import pandas as pd# type: ignore # import everything ml # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader, TensorDataset # pyright: ignore[reportMissingImports]et
import torchvision.transforms as transforms # pyright: ignore[reportMissingImports]
import torchvision.datasets as datasets # pyright: ignore[reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import torch.optim as optim # pyright: ignore[reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import re # pyright: ignore[reportMissingImports]
import string # pyright: ignore[reportMissingImports]
from wordcloud import WordCloud # pyright: ignore[reportMissingImports]
import nltk # pyright: ignore[reportMissingImports]
from collections import Counter # pyright: ignore[reportMissingImports]
import contractions # pyright: ignore[reportMissingImports]
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import math # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset, DataLoader # pyright: ignore[reportMissingImports]
from sklearn.feature_extraction.text import TfidfVectorizer # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import os # pyright: ignore[reportMissingImports]
from PIL import Image, ImageFile # pyright: ignore[reportMissingImports]
import torchvision.models as models # pyright: ignore[reportMissingImports]
from PIL import ImageDraw # pyright: ignore[reportMissingImports]
import datasets # pyright: ignore[reportMissingImports]
from datasets import load_dataset # pyright: ignore[reportMissingImports]
from torch.utils.data import random_split  # pyright: ignore[reportMissingImports]



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

class FakeDataset(Dataset):
    def __init__(self, image, transform=None):
        self.image = image
        self.transform = transform
    def __len__(self):
        return 5000
    def __getitem__(self, idx):
        return self.transform(self.image) if self.transform else self.image
        
            
# Define transform
transform_pipeline = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu") # sue me

dataset = FakeDataset(
    image=Image.open("test_images/benchmark.png"),  # Ensure image is in RGB format
    transform=transform_pipeline
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

image = "test_images/benchmark.png"
heightmap_gen_model = UNet(in_channels=3, out_channels=1, features=[64,128,256,512,1024]).to(device)
terrain_gen_model = UNet(in_channels=3, out_channels=3).to(device)

heightmap_gen_model.load_state_dict(torch.load('../models/turbo_heightmap_unet_model.pth', map_location=device))
terrain_gen_model.load_state_dict(torch.load('../models/turbo_terrain_unet_model.pth', map_location=device))


# def test_single_image(image, model=heightmap_gen_model, secondmodel=terrain_gen_model, device=torch.device("cpu")):

#     image = Image.open(image)
#     # image = image.convert("RGB")  # Ensure image is in RGB format
#     image = transform_pipeline(image)  # Apply the transformation pipeline
#     image = image.to(device)  # Move the image tensor to the specified device
#     # print(image)
    
#     model.eval()
#     secondmodel.eval()

#     # Ensure inputs are batched and on device
#     image = image.unsqueeze(0).to(device)  # shape: [1, C, H, W]

#     with torch.no_grad():
#         print(image)
#         output1 = model(image)
#         print(f"outputs shape: {output1.shape}")
#         output2 = secondmodel(image)
#         print(f"second_outputs shape: {output2.shape}")

#         # Normalize outputs
#         # print(output1.min(), output1.max())
#         # output1 = output1
#         print(output1)
#         # output1 = output1 * 65535.0

#         # Convert to numpy for display
#         input_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
#         output1_np = output1.cpu().squeeze(0).permute(1, 2, 0).numpy()
#         output2_np = output2.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
#         return output1_np, output2_np

def test_models(model=heightmap_gen_model, secondmodel=terrain_gen_model,  dataloader=dataloader, device=torch.device('cpu'),index=0):
    model.eval()
    
    with torch.no_grad():
        for idx, (segmentation) in enumerate(dataloader):
            if(idx != index):
                continue
            images = segmentation
            # terrain_image = terrain  # Use heightmap as the correct image
            # height_image = height  # Use heightmap as the correct image
            images = images.to(device)
            # model, secondmodel = model.to(device), secondmodel.to(device)

            outputs = model(images)
            print(f"outputs shape: {outputs.shape}")
            second_outputs = secondmodel(images)
            print(f"second_outputs shape: {second_outputs.shape}")

            # If using sigmoid or BCE/MSE outputs
            outputs = (outputs/65535.0)
            # outputs = torch.sigmoid(outputs)/65535.0
            # print(f"outputs shape: {outputs.shape}")
            # second_outputs = torch.sigmoid(second_outputs)
            # print(f"second_outputs shape: {second_outputs.shape}")
            # Move to CPU and convert to numpy
            inputs_np = images.cpu().permute(0, 2, 3, 1).numpy()
            outputs_np = outputs.cpu().permute(0, 2, 3, 1).clamp(0,1).numpy()
            second_outputs_np = second_outputs.cpu().permute(0, 2, 3, 1).clamp(0,1).numpy()
            # terrain_image_np = terrain_image.cpu().permute(0, 2, 3, 1).numpy()
            # height_image_np = height_image.cpu().permute(0, 2, 3, 1).numpy()

            for i in range(min(8, images.size(0))):  # Show/save up to 4 samples
                fig, axs = plt.subplots(1, 3, figsize=(8, 4))
                axs[0].imshow(inputs_np[i])
                axs[0].set_title("Input")
                axs[1].imshow(outputs_np[i])
                axs[1].set_title("Height Output")
                axs[2].imshow(second_outputs_np[i])
                axs[2].set_title("Terrain Output")
                # axs[3].imshow(terrain_image_np[i])
                # axs[3].set_title("Terrain Image")
                # axs[4].imshow(height_image_np[i])
                # axs[4].set_title("Heightmap Image")

                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()

                plt.show()

            
            break  # only first batch for quick test
        
# test_models(heightmap_gen_model, terrain_gen_model, test_loader, device, index=10)
test_models()

# if __name__ == "__main__":
#   output1, output2 = test_single_image(image)
#   plt.figure(figsize=(12, 6))
#   plt.subplot(1, 3, 1)
#   plt.imshow(output1)
#   plt.title('Heightmap Output')
#   plt.axis('off')
#   plt.subplot(1, 3, 2)
#   plt.imshow(output2)
#   plt.title('Terrain Output')
#   plt.axis('off')
#   plt.subplot(1, 3, 3)
#   plt.imshow(Image.open(image))
#   plt.title('Input Image')
#   plt.axis('off')
#   plt.tight_layout()
#   plt.show()