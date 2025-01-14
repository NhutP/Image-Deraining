# from utils import *
# import pathlib
# from tqdm import tqdm

# gt = pathlib.Path('./Datasets/test')
# res = pathlib.Path('./results')

# for dataset in gt.iterdir():
#   print(dataset)
#   psnr = 0
#   ssmi = 0

#   target_img = list((dataset / 'gt').iterdir())
  
#   for i in tqdm(target_img):
#     img1 = load_img(i.absolute())
#     img2 = load_img((res / dataset.name / i.name).absolute())   
#     psnr += calculate_psnr(img1, img2)
#     ssmi += calculate_ssim(img1, img2)

#   psnr = psnr / len(target_img)
#   ssmi = ssmi / len(target_img)

#   print(f"PSNR: {psnr}")
#   print((f"SSMI: {ssmi}"))
#   print('-------------------')

import pathlib
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2

# # Function to load an image
# def load_img(img_path):
#     img = cv2.imread(str(img_path))
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# # Function to calculate PSNR (Peak Signal-to-Noise Ratio)
# def calculate_psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100  # No noise, images are identical
#     max_pixel = 255.0
#     return 20 * np.log10(max_pixel / np.sqrt(mse))

# # Function to calculate SSIM (Structural Similarity Index)
# def calculate_ssim(img1, img2):
#     return ssim(img1, img2, multichannel=True)

# Function to load an image
def load_img(img_path):
    img = cv2.imread(str(img_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Function to resize an image to match another image's dimensions
def resize_img(img, target_shape):
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

# Function to calculate PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2):
    # Resize img2 to match img1's dimensions
    img2_resized = resize_img(img2, img1.shape)
    mse = np.mean((img1 - img2_resized) ** 2)
    if mse == 0:
        return 100  # No noise, images are identical
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Function to calculate SSIM (Structural Similarity Index)
def calculate_ssim(img1, img2):
    # Resize img2 to match img1's dimensions
    img2_resized = resize_img(img2, img1.shape)
    return ssim(img1, img2_resized, multichannel=True)

# Paths to dataset
gt = pathlib.Path('./Datasets/test')
res = pathlib.Path('./results')

# Iterate over dataset directories
for dataset in gt.iterdir():
    print(dataset)
    psnr = 0
    ssmi = 0

    target_img = list((dataset / 'gt').iterdir())

    # Iterate through all images in 'gt' folder
    for i in tqdm(target_img):
        img1 = load_img(i.absolute())
        img2 = load_img((res / dataset.name / i.name).absolute())   
        
        psnr += calculate_psnr(img1, img2)
        ssmi += calculate_ssim(img1, img2)

    # Average PSNR and SSIM over the dataset
    psnr = psnr / len(target_img)
    ssmi = ssmi / len(target_img)

    print(f"PSNR: {psnr}")
    print(f"SSIM: {ssmi}")
    print('-------------------')
