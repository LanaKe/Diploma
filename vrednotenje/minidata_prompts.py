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
    def __init__(self, folder_path, json_file, transform=None):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, fname)
            for fname in sorted(os.listdir(folder_path))
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
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
            file_name = os.path.basename(v["file_name"])
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
        return len(self.image_paths)

    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
    
        file_name = image_path
        prompt = self.annotation_pair[file_name]

        return {"img": img, "prompt": prompt}
    '''
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        prompt = self.annotation_pair.get(self.image_paths[idx], "No item category") '''
        
    
