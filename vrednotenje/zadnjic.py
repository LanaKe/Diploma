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
        
        real_name = item['real']
        male_op_name = item['male_op']
        male_dp_name = item['male_dp']
        velike_op_name = item['velike_op']
        velike_dp_name = item['velike_dp']

        image_folder= '/shared/home/lana.kejzar/Diploma/SAM/evalvacija'
        img_folder1 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_op'
        img_folder2 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/velike_dp'
        img_folder3 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_dp'
        img_folder4 = '/shared/home/lana.kejzar/Diploma/finale/vrednotenje/male_op'
    
        #img_name = self.data.iloc[idx, 0]  # First column: image path
        img_path = os.path.join(image_folder, real_name)
        real = Image.open(img_path).convert("RGB")
        img_path = os.path.join(img_folder4, male_op_name)
        male_op = Image.open(img_path).convert("RGB")
        img_path = os.path.join(img_folder3, male_dp_name)
        male_dp = Image.open(img_path).convert("RGB")
        img_path = os.path.join(img_folder1, velike_op_name)
        velike_op = Image.open(img_path).convert("RGB")
        img_path = os.path.join(img_folder2, velike_dp_name)
        velike_dp = Image.open(img_path).convert("RGB")

        if self.transform:
            real = self.transform(real)
            male_op = self.transform(male_op)
            male_dp = self.transform(male_dp)
            velike_op = self.transform(velike_op)
            velike_dp = self.transform(velike_dp)

        #real = real.to(device)
        #male_op = male_op.to(device)
        #target = target.to(device)
        prompt = self.annotation_pair.get(real_name, "No item category")

        return {
            "real": real,
            "male_op": male_op,
            "male_dp": male_dp,
            "velike_op": velike_op,
            "velike_dp": velike_dp,
            "text_prompt": prompt
        }


