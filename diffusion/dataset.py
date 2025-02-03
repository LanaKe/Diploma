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
    def __init__(self, csv_file, transform=None):
        #self.data = pd.read_csv(csv_file)
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        img_name = item['cloth']
        condition_name = item['pose']
        target_name = item['target']
    
        #img_name = self.data.iloc[idx, 0]  # First column: image path
        img_path = os.path.join('/shared/workspace/lrv/DeepBeauty/data/zalando/train/cloth', img_name)
        cloth = cv2.imread(img_path)
        cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)
        cloth = Image.fromarray(cloth)
        #condition_name = self.data.iloc[idx, 1]  # Third column: conditioning image path
        condition_path = os.path.join('/shared/home/lana.kejzar/Diploma/train_op15_pose', condition_name)
        pose = cv2.imread(condition_path)
        pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
        pose = Image.fromarray(pose)
        #target_name = self.data.iloc[idx, 2]  # Second column: target image path
        target_path = os.path.join('/shared/home/lana.kejzar/Diploma/SAM/izluscena_oblacila', target_name)
        target = cv2.imread(target_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = Image.fromarray(target)

        if self.transform:
            cloth = self.transform(cloth)
            target = self.transform(target)
            pose = self.transform(pose)
        
        #cloth = cloth.to(device)
        #pose = pose.to(device)
        #target = target.to(device)
        
        return {
            "image": cloth,
            "target": target,
            "condition_image": pose
        }


