from dataclasses import dataclass
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from diffusers.utils import load_image
from PIL import Image

@dataclass
class TrainingConfig:
    image_size = (352, 256)  # the generated image resolution
    train_batch_size = 2
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 100
    save_model_epochs = 100
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "inferenca4"  # the model name locally and on the HF Hub

    seed = 0


config = TrainingConfig()

from minidata import MyDataset
#from datasett import MyDataset
from torchvision import transforms
import torch
import os

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count())  # Number of GPUs detected

preprocess = transforms.Compose(
    [
        transforms.Resize((352, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

csv_file = "trening.csv"
json_file = "/shared/workspace/lrv/DeepBeauty/data/zalando/vitonhd_train_tagged.json"

dataset = MyDataset(csv_file, json_file, transform=preprocess)
print(f"{len(dataset)=}")
print(type(dataset))
print("Target type:", type(dataset[0]["target"]))
print("Item tag:", type(dataset[0]["text_prompt"]))
print(dataset[0]["text_prompt"])

from testna import MyDataset
eval_data = MyDataset(csv_file)
print(f"{len(eval_data)=}")


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples[0]]
    return {"images": images}


print("model se najprej nauči na 100 slikah in 200 epohah, nato pa generira 10 različnih slik, POZA, na drug način loada slike")
#dataset = transform(dataset)
print(type(dataset))
image = dataset[0]["target"]
print(type(image), image.shape)

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)


from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

#from diffusers import StableDiffusionPipeline

from pipe4 import StableDiffusionPipeline

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


# Load the Stable Diffusion v1.5 pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
unet = pipeline.unet
tokenizer = pipeline.tokenizer
vae = pipeline.vae
text_encoder = pipeline.text_encoder
feature_extractor = pipeline.feature_extractor

print(pipeline)

conv_new = torch.nn.Conv2d(
    in_channels=12,  # Updated to take 8 input channels
    out_channels=unet.conv_in.out_channels,  # Keep the same output channels
    kernel_size=3,
    padding=1,
)


torch.nn.init.kaiming_normal_(conv_new.weight)
conv_new.weight.data = conv_new.weight.data * 0

conv_new.weight.data[:, :4] = unet.conv_in.weight.data
conv_new.bias.data = unet.conv_in.bias.data

unet.conv_in = conv_new  # Replace the old conv layer
unet.config['in_channels'] = 12  # Update the config dictionary
unet.config.in_channels = 12  # Update any other config attribute

print(unet.conv_in)


'''
conv_out_new = torch.nn.Conv2d(
    in_channels=unet.conv_out.in_channels,  # Keep the number of input channels
    out_channels=8,  # Change this to your required number of output channels
    kernel_size=3,
    padding=1,
)
# Initialize weights like the original U-Net
torch.nn.init.kaiming_normal_(conv_out_new.weight)

# Transfer weights from the old layer (assuming previous output was 4 channels)
conv_out_new.weight.data[:4, :, :, :] = unet.conv_out.weight.data[:4, :, :, :]  # Copy first 4 channels

# Set bias (if needed)
conv_out_new.bias.data = unet.conv_out.bias.data

# Replace the old output convolution
unet.conv_out = conv_out_new  # Assign the new layer

# Update U-Net config
unet.config.out_channels = 8  # Ensure the config reflects the change

print(unet.conv_out)  # Verify the new layer '''




from transformers import CLIPFeatureExtractor


#captions = ["t-shirt"] * config.train_batch_size
#captions = {"captions": captions}  # Store in dictionary
#print(captions)
#print(type(captions))
#print(captions.shape)

def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in enumerate(examples[caption]["text_prompt"]):
            if isinstance(caption, str):
                captions.append(caption)
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


#inputs = tokenizer(dataset, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
#print(inputs)



sample_image = dataset[0]["target"].unsqueeze(0)
print("Input shape:", sample_image.shape)


#print("Output shape:", model(sample_image, inputs, timestep=0).sample.shape)

import torch
from PIL import Image
from diffusers import DDPMScheduler


#noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

import torch.nn.functional as F
'''
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise) '''

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers.utils import make_image_grid
from diffusers.utils.torch_utils import randn_tensor
import os
stevec = 0

trsf = transforms.Compose([
    transforms.Resize((352, 256)),
    transforms.ToTensor(),    # Convert to [C, H, W], values in [0,1]
    transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension -> [1, C, H, W]
])


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    global stevec
    while stevec < 10:
        poza = eval_data[stevec]["condition_image"]
        poza_tensor = trsf(poza).to(device)
        #print(f"{poza.shape=}, {poza.type=}")
        #poza = poza.unsqueeze(0).to(device)
        #print(f"{poza.shape=}, {poza.type=}")

        tshirt = eval_data[stevec]["image"]
        tshirt_tensor = trsf(tshirt).to(device)
        #print(f"{tshirt.shape=}, {tshirt.type=}")
        #tshirt = tshirt.unsqueeze(0).to(device)
        #print(f"{tshirt.shape=}, {tshirt.type=}")
        
        #shape = (config.eval_batch_size, 8, 44, 32)
        # #latents = randn_tensor(shape)
        
        images = pipeline(
            prompt="t-shirt, realistic, high quality image",
            guidance_scale=1.0,
            batch_size=config.eval_batch_size,
            condition = poza_tensor,
            start = tshirt_tensor,
            width=256,
            height=352,
            generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
        ).images
        # Make a grid out of the images
        tshirt_pil = transforms.ToPILImage()
        tshirt_pil = tshirt_pil(tshirt_tensor.squeeze(0))
        poza_pil = transforms.ToPILImage()
        poza_pil = poza_pil(poza_tensor.squeeze(0))
        images = images[0]
        #print(type(tshirt_pil))
        #print(type(poza_pil))
        #print(type(images))
        image_grid = make_image_grid([tshirt_pil, poza_pil, images], rows=1, cols=3)
        #image_grid = images[0]
        stevec+=1
        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{stevec}.png")
        print("slika se je shranila")

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

def train_loop(config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["target"]
            poses = batch["condition_image"]
            tshirts = batch["image"]
            prompts = batch["text_prompt"]
            
            #print(f"{poses.type=}")
            #print(type(poses), type(prompts))

            #convert images to latent space
            latents = vae.encode(clean_images.to(dtype=vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            lat_pose = vae.encode(poses.to(dtype=vae.dtype)).latent_dist.sample()
            lat_pose = lat_pose * vae.config.scaling_factor

            lat_tshirt = vae.encode(tshirts.to(dtype=vae.dtype)).latent_dist.sample()
            lat_tshirt = lat_tshirt * vae.config.scaling_factor

            # Sample noise to add to the images
            #noise = torch.randn(clean_images.shape, device=clean_images.device)
            noise = torch.randn_like(latents)
            bs = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device,
            )
            
            inputs = tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
            encoder_hidden_states = text_encoder(**inputs.to(device), return_dict=False)[0]
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Get the text embedding for conditioning
            #print(f"{noisy_latents.device=}")
            #print(f"{encoder_hidden_states.device=}")
            #print(f"{timesteps.device=}")
            #print(f"{noisy_latents.shape=}")
            #print(f"{poses.shape=}")
            #poses_resized = F.interpolate(poses, size=(44, 32), mode="bilinear", align_corners=False)
            #print(f"{lat_pose.shape=}")
            vhod = torch.cat([noisy_latents, lat_pose, lat_tshirt], dim=1)
            #print(f"{vhod.shape=}")

            with accelerator.accumulate(unet):
                # Predict the noise residual
                noise_pred = unet(vhod, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
            #if step%100==0:
                #print("tukaj smo")
        if accelerator.is_main_process:
            #pipeline = StableDiffusionPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),  # Use accelerator.unwrap_model(model) if training
                scheduler=noise_scheduler,
                safety_checker=None,  # Use "None" if you don’t need safety checks
                feature_extractor=feature_extractor
            ).to(device)

            #print("Stable Diffusion pipeline initialized successfully!")

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir)



from datetime import datetime
start_time = datetime.now()
print("zacetek ob", start_time)
#evaluate(config, 0, pipeline)
train_loop(config, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))   