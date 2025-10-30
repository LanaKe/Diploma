import json
import cv2
import numpy as np
import pandas as pd
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, csv_file, json_file, transform=None):
        #self.data = pd.read_csv(csv_file)
        self.data = pd.read_csv(csv_file)
        #self.data = self.data.iloc[:num_images]
        self.transform = transform

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        if isinstance(json_data, dict):  # If JSON is a dictionary
            #print("json is a dict")
            json_data = json_data.get("data", [])

        annotation_list = [
            "colors",
            # "textures",
            "sleeveLength",
            "item",
        ]

        self.annotation_pair = {}
        for v in json_data:
            file_name = v["file_name"]
            for elem in v.get("tag_info", []):
                annotation_str = ""
                for template in annotation_list:
                    for tag in v.get("tag_info", []): 
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                #print(annotation_str)
                self.annotation_pair[v["file_name"]] = annotation_str


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        img_name = item['realne']
    
        #img_name = self.data.iloc[idx, 0]  # First column: image path
        img_path = os.path.join('/shared/workspace/lrv/DeepBeauty/data/zalando/test/cloth', img_name)
        cloth = Image.open(img_path).convert("RGB")

    
        if self.transform:
            cloth = self.transform(cloth)
    
        prompt = self.annotation_pair.get(img_name, "No item category")

        return {
            "image": cloth,
            "text_prompt" : prompt   
        }


file = "labels.csv"
json_file = "/shared/workspace/lrv/DeepBeauty/data/zalando/vitonhd_test_tagged.json"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = MyDataset(file, json_file, transform=transform)

print(len(dataset))

import csv


csv_file = "captions.csv"

with open(csv_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for i in range(len(dataset)):
        caption = dataset[i]["text_prompt"]
        writer.writerow([caption])  # write one string per row
