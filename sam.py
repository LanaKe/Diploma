'''
!nvidia-smi

import os
HOME = os.getcwd()
print("HOME:", HOME)

!pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'¸

!pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision==0.23.0

!mkdir -p {HOME}/weights
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights
'''
import os
from datasets import Dataset
from PIL import Image
import cv2
import supervision as sv
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
HOME = os.getcwd()

CHECKPOINT_PATH = os.path.join(HOME, "SAM", "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

#!pip install datasets

images_folder = '/shared/workspace/lrv/DeepBeauty/data/zalando/train/image'
masks_folder = '/shared/workspace/lrv/DeepBeauty/data/zalando/train/gt_cloth_warped_mask'
print(images_folder)
print(masks_folder)

# Function to load images from a folder and combine them into a numpy array
def load_image_stack(folder_path):
    # List and sort files in the folder
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
    image_stack = []

    # Load each image
    for file_name in file_list:
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)
        image_array = np.array(image)  # Convert to numpy array
        image_stack.append(image_array)
      
    # Stack images along a new dimension (e.g., depth)
    return np.stack(image_stack, axis=0)

# Load the images and masks
large_images = load_image_stack(images_folder)
large_masks = load_image_stack(masks_folder)
print("loaded")
print("Images shape:", large_images.shape)
print("Masks shape:", large_masks.shape)

# Convert the NumPy arrays to Pillow images and store them in a dictionary
dataset_dict = {
    "image": [Image.fromarray(img) for img in large_images],
    "mask": [Image.fromarray(mask) for mask in large_masks],
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)
#print(dataset)
'''
count = 0
for example in dataset:
    gt = np.array(example['mask'])
    if np.all(gt == 0): 
        count = count + 1

print(count) 
'''

#example = dataset[509]
#print(np.array(example['image']).shape)
#example['image'].save('slika.png')
#print("prisli do tukaj")
for idx, example in enumerate(dataset):
#idx = 1
#for example in dataset:
    image_rgb = np.array(example['image'])
    ground_truth_seg = np.array(example['mask'])
    if np.all(ground_truth_seg == 0): 
        print(idx)
        continue
        
    #image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    sam_result = mask_generator.generate(image_rgb)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections.from_sam(sam_result=sam_result)

    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    masks = [
    mask['segmentation']
    for mask
    in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]

    mask_predictor = SamPredictor(sam)
    #uvozena = np.array(example['mask'])
    #uvozena = cv2.cvtColor(uvozena, cv2.COLOR_RGB2GRAY)

    #ground_truth_seg = np.array(example['mask'])
    #ground_truth_seg.save('maska11.png')
    #print(ground_truth_seg.shape)
    #uvozena = cv2.cvtColor(uvozena, cv2.COLOR_RGB2GRAY)
    #uvozena = cv2.cvtColor(uvozena, cv2.COLOR_BGR2GRAY)
    
    def get_bounding_box(ground_truth_map):
    # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox


    input_boxes = get_bounding_box(ground_truth_seg)
    input_boxes

    #print("input boxes print:", input_boxes)
    box = np.array(input_boxes)
    #print("tole je pa samo en box:", box)

    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=False
    )

    box_annotator = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    #print("druga maska")

    bm = Image.fromarray(masks[0])
    binary_mask = bm.point(lambda p: 1 if p > 0 else 0)

    #plt.imshow(binary_mask)

    image = image_rgb

    image_array = np.array(image)
    binary_mask_array = np.array(binary_mask)

    # Apply the mask: Set the areas where the mask is 0 (black) to 0 (black) in the image
    masked_image_array = image_array.copy()
    masked_image_array[binary_mask_array == 0] = 255  # Set masked areas to black

    # Convert the masked image back to a PIL image
    masked_image = Image.fromarray(masked_image_array)
    #masked_image.save('maska.png')
    masked_image.save(f'SAM/izluscena_oblacila/tshirt_{idx}.png')

print("finito")

