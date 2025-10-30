# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

import os


import torch
import cv2
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from diffusers.utils import load_image, make_image_grid
from PIL import Image

import pandas as pd


from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torchvision import transforms, datasets
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def load_images_from_folder(folder, target_shape):
    images = []
    for filename in os.listdir(folder):
        # Load image and convert to RGB
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=target_shape)  # Resize while loading
        img_array = img_to_array(img)  # Convert to array
        images.append(img_array)
    return np.array(images)


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
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

kid = KernelInceptionDistance(subset_size=50)

# Path to folder containing 100 images
image_folder = '/shared/home/lana.kejzar/Diploma/SAM/izluscena_oblacila'
images = load_images_from_folder(image_folder, target_shape=(299, 299))
# Convert to float32
images = images.astype('float32')
print('Prepared originalne slike', images.shape)

image_folder = '/shared/home/lana.kejzar/Diploma/finale/rezultati_testdensepose_quick/samples'
densepose = load_images_from_folder(image_folder, target_shape=(299, 299))
densepose = densepose.astype('float32')
print('Prepared densepose slike', densepose.shape)


image_folder = '/shared/home/lana.kejzar/Diploma/finale/rezultati_samoslike/samples'
openpose = load_images_from_folder(image_folder, target_shape=(299, 299))
openpose = openpose.astype('float32')
print('Prepared openpose slike', openpose.shape)

kid.update(images, real=True)
kid.update(densepose, real=False)

# Compute final KID score
kid_mean, kid_std = kid.compute()
print(f"KID za densepose Mean: {kid_mean.item():.5f}, Std: {kid_std.item():.5f}")

kid.update(images, real=True)
kid.update(openpose, real=False)

# Compute final KID score
kid_mean, kid_std = kid.compute()
print(f"KID za openpose Mean: {kid_mean.item():.5f}, Std: {kid_std.item():.5f}")




'''
# Preprocess for InceptionV3
images = preprocess_input(images)
densepose = preprocess_input(densepose)
openpose = preprocess_input(openpose)
print('Prepared procesirane', images.shape, densepose.shape, openpose.shape)

fid = calculate_fid(model, images, densepose)
print('Izračunan FID med originalnimi slikami in z densepose naučenim modelom: %.3f' % fid)

#fid = calculate_fid(model, openpose, openpose)
#print('FID za OP in OP: %.3f' % fid)

# fid between images1 and images2
fid = calculate_fid(model, images, openpose)
print('Izračunan FID med originalnimi slikami in z openpose naučenim modelom: %.3f' % fid)

#fid = calculate_fid(model, op15, op15)
#print('FID za OP15 in op15: %.3f' % fid)





# define two fake collections of images
images1 = randint(0,255, 10*32*32*3)
images1 = images1.reshape((10,32,32,3))
images2 = randint(0,255, 10*32*32*3)
images2 = images2.reshape((10,32,32,3))
print('Prepared images1 in images2', images1.shape, images2.shape)
# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled images1 in images2', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# fid between images1 and images1
fid = calculate_fid(model, images1, images1)
print('FID (same) za images1 in images1: %.3f' % fid)
# fid between images1 and images2
fid = calculate_fid(model, images1, images2)
print('FID (different) za images1 in images2: %.3f' % fid) '''


#fid = calculate_fid(model, images, images)
#print('FID (same) za images in images: %.3f' % fid)

