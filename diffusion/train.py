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
import random
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline
from diffusers import DDPMScheduler
from transformers import AutoTokenizer
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline 
from diffusers import StableDiffusionImg2ImgPipeline

from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/test3')


@dataclass
class TrainingConfig:
    #image_size = 128  # the generated image resolution
    train_batch_size = 2
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "test10"  # the model name locally and on the HF Hub
    seed = 0


config = TrainingConfig()
 
from dataset import MyDataset
transform = transforms.Compose([
    transforms.Resize((704, 512)),  # Resize all images to 512x512
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

csv_file = "trening.csv"
dataset = MyDataset(csv_file, transform=transform)
print(len(dataset))


item = dataset[1234]
image = item['image']
pose = item['condition_image']
target = item['target']
print(image.shape)
print(pose.shape)
print(target.shape)
print(type(image))


dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=[704,512],  # the target image resolution
    in_channels=9,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

#concatenated = torch.cat((image, pose), dim=0)
#combined = torch.cat((image.unsqueeze(0), pose.unsqueeze(0)), dim=1)  # Shape: [1, 6, 704, 512]
input = torch.cat((image.unsqueeze(0), pose.unsqueeze(0), target.unsqueeze(0)), dim=1)

print("Input shape:", input.shape)

print("Output shape:", model(input, timestep=0).sample.shape)

import torch
from PIL import Image
from diffusers import DDPMScheduler

import torch.nn.functional as F

noise_scheduler = DDPMScheduler(num_train_timesteps=3000)
# noise = torch.randn(input.shape)
# timesteps = torch.LongTensor([50])
# noisy_image = noise_scheduler.add_noise(input, noise, timesteps)
# #print("noisy image shape:", noisy_image)


# noise_pred = model(noisy_image, timesteps).sample
# print(noise_pred.shape)
# print(noise.shape)
# loss = F.mse_loss(noise_pred, noise)
# print("Loss:", loss)

from diffusers.optimization import get_cosine_schedule_with_warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)

#from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

stevec = 0

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    tshirt = Image.open("tshirt.jpg").convert("RGB")
    tshirt = transform(tshirt)
    pose = Image.open("pose.png")
    pose = transform(pose)
    #print("type of tshirt: ", type(tshirt), tshirt.shape)
    #print("type of pose: ", type(pose), pose.shape)
    images = pipeline(
        pose = pose,
        tshirt = tshirt,
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    #print(type(images))
    #images.save('slika.png')
    global stevec
    stevec+=1
    images.save(f'test422/output_{stevec}.png')
    
    ''' Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png") '''


from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from pipeline import DDPMPipeline

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            cloth = batch['image']
            pose = batch['condition_image']
            target = batch['target']

            # print(f"{target.device=}")
            # print(f"{cloth.device=}")
            # print(f"{pose.device=}")

            # Target is the thing that needs to be denoised
            # So we should only be progressively adding more noise to target
            
            # Sample noise to add to the images
            #print("target putput:", type(target), target.shape)
            noise = torch.randn(target.shape, device=target.device)
            bs = target.shape[0]   # target has same bs as final_input
            #print("noise image putput:", type(noise), noise.shape)
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=target.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_targets = noise_scheduler.add_noise(target, noise, timesteps)
            #print("noisy image putput:", type(noisy_targets), noisy_targets.shape)

            # print(f"{noisy_targets.shape=}")
            # print(f"{cloth.shape=}")
            # print(f"{pose.shape=}")

            # print(f"{noisy_targets.device=}")
            # print(f"{cloth.device=}")
            # print(f"{pose.device=}")

            # Make the final input tensor
            final_input = torch.cat((cloth, pose, noisy_targets), dim=1)
            
            #print(f"{final_input.shape=}")
            
            # import sys
            # sys.exit(0)
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(final_input, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            writer.add_scalar('training loss', loss, step)

            
            if step%100==0:
                #print(f"Epoch {epoch+1}, Loss: {loss.item()}")
                #print(f"loss: {loss.detach().item()}, lr: {lr_scheduler.get_last_lr()[0]}, step: {global_step}")
                #f.write(f"loss: {loss.detach().item()}, lr: {lr_scheduler.get_last_lr()[0]}, step: {global_step}")
                # After each epoch you optionally sample some demo images with evaluate() and save the model
                if accelerator.is_main_process:
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                    if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                        evaluate(config, epoch, pipeline)

                    if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                        pipeline.save_pretrained(config.output_dir)

    
        


from datetime import datetime
start_time = datetime.now()
print("zacetek ob", start_time)
train_loop(config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
