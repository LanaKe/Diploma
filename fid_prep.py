import os

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

from huggingface_hub import HfApi
from pathlib import Path
from controlnet_aux import OpenposeDetector


import albumentations as A

augumentations = A.Compose([
    #A.Resize(256, 256),  # Example: Resize images to 256x256
    A.HorizontalFlip(p=0.0),  # Example: Random horizontal flip
], additional_targets={'image': 'image'})

def tensortoimg(image):
    image = image.permute(1,2,0).numpy()
    image = Image.fromarray((image * 255).astype(np.uint8))

    return image

class CustomDataset(Dataset):
  def __init__(self, df):
    self.df = pd.read_csv(df)
    self.augumentations = augumentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]
    image_name = row.image

    image_path = os.path.join('/shared/workspace/lrv/DeepBeauty/data/zalando/train/image', image_name)
    #print(image_path)
    #image_path = os.path.join('zip/image', self.df.iloc[idx, 0])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   
    aug = self.augumentations(image=img)
    img = aug['image']
    #(h,w,c) -> (c,h,w)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    img = torch.Tensor(img) / 255.0

    return img


trainset = CustomDataset("images.csv")
#print(trainset[0][0].shape)
#print(trainset[0][1].shape)
print(f"Size of trainset: {len(trainset)}")


trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

print(f"total no. of batches: {len(trainloader)}")

for image in trainloader:
  print(image.shape)
  print(type(image))
  break


############################# 

checkpoint = "lllyasviel/control_v11p_sd15_openpose"
controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
generator = torch.manual_seed(0)

#izlušči poze za openpose_v11_sd15
for idx, image in enumerate(trainset):
  #image = trainset[1]
  #print(type(image))
  arr = image.permute(1,2,0).numpy()
  arr = Image.fromarray((arr * 255).astype(np.uint8))
  #print(type(arr))  # Output: (256, 256, 3)
  control_image = processor(arr, hand_and_face=True)
  control_image.save(f'train_op15_pose/output_{idx}.png')
  #if idx==5: break

print("končane poze")

'''
##############################OPENPOSE_SD15
for idx, image in enumerate(trainset):
  #image = trainset[1]
  #print(type(image))
  arr = image.permute(1,2,0).numpy()
  arr = Image.fromarray((arr * 255).astype(np.uint8))
  #print(type(arr))  # Output: (256, 256, 3)
  control_image = processor(arr, hand_and_face=True)
  output1 = pipe("woman in a black shirt", control_image, num_inference_steps=30, generator=generator).images[0]
  output1.save(f'op15_res/output_{idx}.png')
  #print("tudi do tu")
  #if idx == 20: break


print("Zakljucen openpose15")

####################################OPENPOSE
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

for idx, image in enumerate(trainset):
  #image = trainset[1]
  #print(type(image))
  arr = image.permute(1,2,0).numpy()
  arr = Image.fromarray((arr * 255).astype(np.uint8))
  control = processor(arr)
  #print(type(arr))  # Output: (256, 256, 3)

  output2 = pipe("woman in a black shirt", control, num_inference_steps=30).images[0]
  output2.save(f'op_res/output_{idx}.png')
  #print("tudi do tu")
  #if idx == 5: break

print("Finito") '''
  