import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv

import dataset
print(dir(dataset))

from dataset import MyDataSet 


def pripravi_dataset(folder_path, captions, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = MyDataSet(folder_path, captions, transform=transform)
    print("Dataset created with", len(dataset), "images.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

prompts = []

with open("captions.csv", mode="r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prompt = row["prompt"].strip()
        if prompt:
            prompts.append(prompt)

print(len(prompts), "prompts loaded from captions.csv")
print(prompts[:20])  # Print first 5 prompts for verification
captions = prompts
image_folder= '/shared/home/lana.kejzar/Diploma/SAM/evalvacija_mini'
#image_folder = '/shared/home/lana.kejzar/Diploma/image/test'
img_folder1 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_op'
img_folder2 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_dp'
img_folder3 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_dp'
img_folder4 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_op'

real_images = pripravi_dataset(image_folder, captions, batch_size=16)
print("Real images loaded", len(real_images))  # Should be (N, 3, 299, 299)
velike_op = pripravi_dataset(img_folder1, captions, batch_size=16)
velike_dp = pripravi_dataset(img_folder2, captions, batch_size=16)
male_dp = pripravi_dataset(img_folder3, captions, batch_size=16)
male_op = pripravi_dataset(img_folder4, captions, batch_size=16)

testni_folder = '/shared/home/lana.kejzar/Diploma/image/test'
#real_images = load_images_as_tensor(testni_folder).to(torch.uint8)

print("naložene so, začnimo")
array_folders = [velike_op, velike_dp, male_dp, male_op]

from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
total_score = 0.0
total_images = 0
'''
for batch_images, batch_captions in real_images:
    batch_size = batch_images.size(0)
    score = metric(batch_images, list(batch_captions))  # if captions are strings
    total_score += score.item() * batch_size
    total_images += batch_size
average_score = total_score / total_images
print(f"Average CLIP Score over {total_images} images: {average_score:.4f}") '''

for folder in array_folders:
    total_score = 0.0
    total_images = 0
    for batch_images, batch_captions in folder:
        batch_size = batch_images.size(0)
        score = metric(batch_images, list(batch_captions))  # if captions are strings
        total_score += score.item() * batch_size
        total_images += batch_size
    average_score = total_score / total_images
    print(f"Average CLIP Score over Folder {total_images} images: {average_score:.4f}")

'''
for batch in real_images:
    print(batch.shape)
    score = metric(batch, prompts)
    print("Real images - CLIP Score:", score.detach().round().item())

for folder in array_folders:
    for batch in folder:
        print(batch.shape)
        score = metric(batch, prompts)
        print("Folder - CLIP Score:", score.detach().round().item()) '''

