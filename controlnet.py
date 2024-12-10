from google.colab import files
uploaded = files.upload()

import zipfile
import os

# Replace 'your_file.zip' with the uploaded zip filename
zip_file_name = 'zip.zip'

# Unzip the file
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall('/content')  # Extract to the /content directory

import torch
import cv2
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from diffusers.utils import load_image, make_image_grid
from PIL import Image

import pandas as pd


from sklearn.model_selection import train_test_split
from tqdm import tqdm

#!pip install albumentations
#!pip install -q diffusers transformers accelerate opencv-python
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline


import albumentations as A

augumentations = A.Compose([
    #A.Resize(256, 256),  # Example: Resize images to 256x256
    A.HorizontalFlip(p=0.0),  # Example: Random horizontal flip
], additional_targets={'openpose': 'image', 'cloth': 'image'})



class CustomDataset(Dataset):
  def __init__(self, df):
    self.df = pd.read_csv(df)
    self.augumentations = augumentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]
    image_name = row.image


    image_path = os.path.join('zip/image', image_name)
    #image_path = os.path.join('zip/image', self.df.iloc[idx, 0])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    openpose_path = os.path.join('zip/openpose', self.df.iloc[idx, 1])
    openpose = cv2.imread(openpose_path)
    openpose = cv2.cvtColor(openpose, cv2.COLOR_BGR2RGB)

    cloth_path = os.path.join('zip/cloth', self.df.iloc[idx, 2])
    cloth = cv2.imread(cloth_path)
    cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)


    #if self.augumentations:
     # aug = self.augumentations(image=image, body = body)
     # image = aug['image']
      #body = aug['body']  #returns mask and imagea s dictionary format, zato separatamo

    aug = self.augumentations(image=image, openpose=openpose, cloth=cloth)
    image = aug['image']
    openpose = aug['openpose']
    cloth = aug['cloth']
    #(h,w,c) -> (c,h,w)
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    openpose = np.transpose(openpose, (2, 0, 1)).astype(np.float32)
    cloth = np.transpose(cloth, (2, 0, 1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    openpose = torch.round(torch.Tensor(openpose) / 255.0)
    cloth = torch.Tensor(cloth) / 255.0



    return image, openpose, cloth


trainset = CustomDataset("zip/data.csv")
#print(trainset[0][0].shape)
#print(trainset[0][1].shape)
print(f"Size of trainset: {len(trainset)}")
#trainset.__getitem__(0)

trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

print(f"total no. of batches: {len(trainloader)}")

for image, openpose, cloth in trainloader:
  print(image.shape)
  print(openpose.shape)
  print(cloth.shape)
  break

def grid_plot(image, openpose, cloth):

  f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

  ax1.set_title('IMAGE')
  ax1.imshow(image)

  ax2.set_title('OPENPOSE')
  ax2.imshow(openpose)

  ax3.set_title('CLOTH')
  ax3.imshow(cloth)

  f.subplots_adjust(wspace=0.1)
