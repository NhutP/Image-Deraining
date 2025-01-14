from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFilter
import os

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx
import cv2
import io


device = torch.device('cuda' if os.getenv('device') == 'cuda' else 'cpu')

app = FastAPI()

yaml_file = 'Options/Deraining_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g']).to(device)

checkpoint = torch.load('./pretrained_models/deraining.pth', map_location=device)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", './pretrained_models/deraining.pth')
model_restoration.to(device)
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

print('call')
factor=8

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # Load the uploaded image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = np.float32(image)/255.
    img = torch.from_numpy(img).permute(2,0,1)
    

    with torch.no_grad():
    # # Convert the processed image to bytes
    #   torch.cuda.ipc_collect()
    #   torch.cuda.empty_cache()
      input_ = img.unsqueeze(0).to(device)
      h,w = input_.shape[2], input_.shape[3]
      H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
      padh = H-h if h%factor!=0 else 0
      padw = W-w if w%factor!=0 else 0
      input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

      restored = model_restoration(input_)

      # Unpad images to original dimensions
      restored = restored[:,:,:h,:w]

      restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
      
      # convert to BGR to RGB
      restored = restored[..., [2, 1, 0]]

    processed_image = img_as_ubyte(restored)
    processed_image = Image.fromarray(processed_image)# .resize(image.size)
    
    img_bytes = io.BytesIO()
    processed_image.save(img_bytes, format='png')
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type=f"image/png")
