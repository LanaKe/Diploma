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

class Prompts:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        if isinstance(json_data, dict):
            json_data = json_data.get("data", [])

        json_data = json_data[:26]

        annotation_list = [
            "colors",
            "sleeveLength",
            "item",
        ]

        self.annotation_pair = {}
        for v in json_data:
            annotation_str = ""
            for template in annotation_list:
                for tag in v.get("tag_info", []):
                    if tag["tag_name"] == template and tag["tag_category"] is not None:
                        annotation_str += tag["tag_category"] + " "
            self.annotation_pair[v["file_name"]] = annotation_str.strip()
    
    def get_prompt(self, key):
        if isinstance(key, int):
            keys = list(self.annotation_pair.keys())
            if 0 <= key < len(keys):
                file_name = keys[key]
                return self.annotation_pair[file_name]
            else:
                return "Index out of range"
        else:
            return self.annotation_pair.get(key, "No prompt available")
