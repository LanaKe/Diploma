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
    train_batch_size = 4
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 20
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "original"  # the model name locally and on the HF Hub

    seed = 0


config = TrainingConfig()

from datasett import MyDataset
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
dataset = MyDataset(csv_file, transform=preprocess)
print(f"{len(dataset)=}")
print(type(dataset))
print("Target type:", type(dataset[0]["image"]))


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples[0]]
    return {"images": images}


print("PRVI POSKUS S SD")
#dataset = transform(dataset)
print(type(dataset))
image = dataset[0]["image"]
print(type(image), image.shape)

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)


from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion v1.5 pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# Save components for training
pipeline.vae.save_pretrained("./sd15_vae")
pipeline.unet.save_pretrained("./sd15_unet")

print("Stable Diffusion 1.5 U-Net and VAE downloaded and saved!")

tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

# Save locally for reuse
tokenizer.save_pretrained("./sd15_tokenizer")
text_encoder.save_pretrained("./sd15_text_encoder")

print("Tokenizer and text encoder downloaded and saved!")

# Load U-Net and VAE from saved models
model = UNet2DConditionModel.from_pretrained("./sd15_unet", subfolder="vae").to(device)
vae = AutoencoderKL.from_pretrained("./sd15_vae",  subfolder="unet").to(device)

print("U-Net and VAE successfully loaded for training!")

tokenizer = CLIPTokenizer.from_pretrained("./sd15_tokenizer")
text_encoder = CLIPTextModel.from_pretrained("./sd15_text_encoder").to(device)

print("Tokenizer and text encoder successfully loaded!")

from transformers import CLIPFeatureExtractor
feature_extractor = CLIPFeatureExtractor.from_pretrained(model_id, subfolder="feature_extractor")

captions = ["t-shirt"] * config.train_batch_size
#captions = {"captions": captions}  # Store in dictionary
print(captions)
print(type(captions))
#print(captions.shape)
inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
#print(inputs)



sample_image = dataset[0]["image"].unsqueeze(0)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

from diffusers.utils import make_image_grid
import os
stevec = 0

def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    eval = load_image("/shared/home/lana.kejzar/Diploma/stable_diffusion/tshirt.jpg")
    eval = eval.resize((352,256))

    images = pipeline(
        prompt="t-shirt, realistic, high quality image",
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    # Make a grid out of the images
    #image_grid = make_image_grid(images, rows=1, cols=1)
    image_grid = images[0]
    global stevec
    stevec+=1
    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{stevec}.png")
    print("slika se je shranila")
    '''
    act1 = numpy.array(eval)
    act2 = numpy.array(image_grid)
    print(act1.shape, act2.shape)
    #act1 = act1.reshape((352,256))
    #act2 = act2.reshape((352,256))
    fid = calculate_fid(act1, act2)
    print('FID (different): %.3f' % fid)
    '''

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, encoder_hidden_states):
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
            clean_images = batch["image"]

            #convert images to latent space
            latents = vae.encode(clean_images.to(dtype=vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise to add to the images
            #noise = torch.randn(clean_images.shape, device=clean_images.device)
            noise = torch.randn_like(latents)
            bs = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device,
            )
            
            encoder_hidden_states = text_encoder(**inputs.to(device), return_dict=False)[0]
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            # Get the text embedding for conditioning
            #print(f"{noisy_latents.device=}")
            #print(f"{encoder_hidden_states.device=}")
            #print(f"{timesteps.device=}")

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
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

        # After each epoch you optionally sample some demo images with evaluate() and save the model
            #if step%100==0:
        if accelerator.is_main_process:
            #pipeline = StableDiffusionPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            pipeline = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(model),  # Use accelerator.unwrap_model(model) if training
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
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, inputs)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))   