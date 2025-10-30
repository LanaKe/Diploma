import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def load_images_as_tensor(folder_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to InceptionV3 input size
        transforms.ToTensor(),          # Converts to float tensor in [0, 1]
    ])

    image_tensors = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img)  # (3, 299, 299)
            image_tensors.append(tensor)

    if not image_tensors:
        raise ValueError(f"No images found in folder: {folder_path}")

    return torch.stack(image_tensors)  # Shape: (N, 3, 299, 299)

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
#image_folder = '/shared/workspace/lrv/DeepBeauty/data/zalando/train/cloth'
#image_folder = '/shared/home/lana.kejzar/Diploma/SAM/izluscena_oblacila'
#img_folder1 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_op'
img_folder1 = '/shared/home/lana.kejzar/Diploma/finale/rezultati_ucnadensepose/samples'
img_folder2 = '/shared/home/lana.kejzar/Diploma/finale/rezultati_ucnaslike/samples'
#img_folder2 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_dp'
img_folder3 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_dp'
img_folder4 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_op'

real_images = pripravi_dataset(image_folder, batch_size=16)
print("Real images loaded", len(real_images))  # Should be (N, 3, 299, 299)
velike_op = pripravi_dataset(img_folder1, batch_size=16)
velike_dp = pripravi_dataset(img_folder2, batch_size=16)
#male_dp = pripravi_dataset(img_folder3, batch_size=16)
#male_op = pripravi_dataset(img_folder4, batch_size=16)

testni_folder = '/shared/home/lana.kejzar/Diploma/image/test'
#real_images = load_images_as_tensor(testni_folder).to(torch.uint8)

print("naložene so, začnimo")

#velike_op, velike_dp, male_dp, male_op = real_images, real_images, real_images, real_images

array_folders = [velike_op, velike_dp]

# Calculate FID for each set of images
import torch
from torchmetrics.image.fid import FrechetInceptionDistance 

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=2048)

for i in range(len(array_folders)):
    for real_batch in real_images:
        real_batch = (real_batch * 255).to(torch.uint8)
        fid.update(real_batch, real=True)
    #print(real_images.shape)  # torch.Size([N, 3, 299, 299])
    images = array_folders[i]
    #print(images.shape)
    for fake_batch in images:
        fake_batch = (fake_batch * 255).to(torch.uint8)
        fid.update(fake_batch, real=False)
    
    fid_score = fid.compute().item()
    print(f"FID score: {fid_score}")
    fid.reset()  # Reset for next computation
print("zaključili smo FID")

from torchmetrics.image.kid import KernelInceptionDistance
kid = KernelInceptionDistance(subset_size=10)
# generate two slightly overlapping image intensity distributions
for i in range(len(array_folders)):
    for real_batch in real_images:
        real_batch = (real_batch * 255).to(torch.uint8)
        kid.update(real_batch, real=True)
    #print(real_images.shape)  # torch.Size([N, 3, 299, 299])
    images = array_folders[i]
    #print(images.shape)
    for fake_batch in images:
        fake_batch = (fake_batch * 255).to(torch.uint8)
        kid.update(fake_batch, real=False)

    kid_score = kid.compute()
    print(f"KID score: {kid_score}")
    kid.reset()  # Reset for next computation
print("zaključili smo KID")


'''
from torchmetrics.multimodal import CLIPImageQualityAssessment
#metric = CLIPImageQualityAssessment()
metric = CLIPImageQualityAssessment()
all_scores = []

for batch in real_images:
    batch = (batch * 255).to(torch.uint8)  # Required by CLIP IQA
    score = metric(batch)
    #print(score.shape)
    #print(score)  # Should be (N, 1)
    all_scores.append(score)
#all_scores = torch.cat(all_scores, dim=0)
print(len(all_scores))  # Should be (N, 1)
print(all_scores)
average = all_scores.mean().item()
print("CLIPimage q a module je", average)
for imgs in array_folders:
    all_scores = []
    for batch in imgs:
        batch = (batch * 255).to(torch.uint8)  # Required by CLIP IQA
        score = metric(batch)
        all_scores.append(score)
    #all_scores = torch.cat(all_scores, dim=0)
    average = all_scores.mean().item()
print(f"CLIP Image Quality Assessment score MODULE: povprečje je {average}")

print("zaključili smo CLIPimage q a module")




def pripravi_prompts(image_folder, json_file, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    dataset = MyDataset(image_folder, json_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader
print("CLIP score bo narjen samo za module, menda je boljši")
from test import MyDataset



json_file = "/shared/workspace/lrv/DeepBeauty/data/zalando/vitonhd_test_tagged.json"
#csv_file = "evalvacija.csv"
#data = pripravi_prompts(csv_file, json_file)

real_images = pripravi_prompts(image_folder, json_file, batch_size=16)
print("Real images loaded", len(real_images))  # Should be (N, 3, 299, 299)
velike_op = pripravi_prompts(img_folder1, json_file, batch_size=16)
velike_dp = pripravi_prompts(img_folder2, json_file, batch_size=16)
male_dp = pripravi_prompts(img_folder3, json_file, batch_size=16)
male_op = pripravi_prompts(img_folder4, json_file, batch_size=16)


print("naložene so, začnimo")

from torchmetrics.multimodal.clip_score import CLIPScore

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images):
    all_scores = []
    for batch in images:
        img = batch['img']
        prompts = batch['prompt']

        #batch = (batch * 255).clamp(0, 255).to(torch.uint8)

        if img.dtype != torch.uint8:
            img = (img * 255).clamp(0, 255).to(torch.uint8)

        for slika, pro in zip(img, prompts):
            score = metric(slika, pro).detach()
            all_scores.append(score)
            #print(f"CLIP score for image: {score.item()}, prompt: '{pro}'")

    all_scores = torch.stack(all_scores)
    average = all_scores.mean().item()
    #print(f"Average CLIPScore: {round(average, 4)}")
    return average


sd_clip_score = calculate_clip_score(real_images)
print(f"CLIP score real images: {sd_clip_score}")

sd_clip_score = calculate_clip_score(velike_op)
print(f"CLIP score velike_op: {sd_clip_score}")

sd_clip_score = calculate_clip_score(velike_dp)
print(f"CLIP score velike_dp: {sd_clip_score}")

sd_clip_score = calculate_clip_score(male_dp)
print(f"CLIP score male_dp: {sd_clip_score}")

sd_clip_score = calculate_clip_score(male_op)
print(f"CLIP score male_op: {sd_clip_score}")

print("končali vse") '''

