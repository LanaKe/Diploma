import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from loading import ImageFolderDataset


def pripravi_dataset(folder_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageFolderDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

image_folder= '/shared/home/lana.kejzar/Diploma/SAM/evalvacija'
#image_folder = '/shared/home/lana.kejzar/Diploma/image/test'
img_folder1 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_op'
img_folder2 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_dp'
img_folder3 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_dp'
img_folder4 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_op'

real_images = pripravi_dataset(image_folder, batch_size=2032)
print("Real images loaded", len(real_images))  # Should be (N, 3, 299, 299)

velike_op = pripravi_dataset(img_folder1, batch_size=2032)
velike_dp = pripravi_dataset(img_folder2, batch_size=2032)
male_dp = pripravi_dataset(img_folder3, batch_size=2032)
male_op = pripravi_dataset(img_folder4, batch_size=2032)
testni_folder = '/shared/home/lana.kejzar/Diploma/image/test'
#real_images = load_images_as_tensor(testni_folder).to(torch.uint8)

print("naložene so, začnimo")

#velike_op, velike_dp, male_dp, male_op = real_images, real_images, real_images, real_images

array_folders = [velike_op, velike_dp, male_dp, male_op]

from torch import randint
from torchmetrics.multimodal import CLIPImageQualityAssessment
metric = CLIPImageQualityAssessment(prompts=(("High quality T-shirt image.", "Low quality T-shirt image."), "quality", "sharpness", "noisiness"),)
for batch in real_images:
    score = metric(batch)
    custom = score['user_defined_0']
    quality = score['quality']
    sharpness = score['sharpness']
    noisiness = score['noisiness']
    print("Real images - Quality:", quality.mean().item(), "Custom:", custom.mean().item(), "Sharpness:", sharpness.mean().item(), "Noisiness:", noisiness.mean().item())
for folder in array_folders:
    for batch in folder:
        score = metric(batch)
        custom = score['user_defined_0']
        quality = score['quality']
        sharpness = score['sharpness']
        noisiness = score['noisiness']
    print (f"Folder - Quality: {quality.mean().item()}, Custom: {custom.mean().item()}, Sharpness: {sharpness.mean().item()}, Noisiness: {noisiness.mean().item()}")
