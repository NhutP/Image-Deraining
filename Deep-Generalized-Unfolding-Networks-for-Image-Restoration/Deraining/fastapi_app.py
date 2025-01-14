from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import io
import torch
from DGUNet import DGUNet
import utils
import torch.nn as nn
from skimage import img_as_ubyte
import os

device = torch.device('cuda' if os.getenv('device') == 'cuda' else 'cpu')

app = FastAPI()

model_restoration = DGUNet()
utils.load_checkpoint(model_restoration,'./pretrained_models/DGUNet.pth')
print("===>Testing using weights: ", './pretrained_models/DGUNet.pth')
# utils.load_checkpoint(model_restoration, os.getenv('ckp_dir') + '/DGUNet.pth')
# print("===>Testing using weights: ", os.getenv('ckp_dir') + '/DGUNet.pth')
model_restoration = model_restoration.to(device)
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

print('call')

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # Load the uploaded image
    image = Image.open(file.file)

    inp = TF.to_tensor(image.resize((720, 480)))

    with torch.no_grad():
    # # Convert the processed image to bytes
    #   torch.cuda.ipc_collect()
    #   torch.cuda.empty_cache()
      # input_    = inp.unsqueeze(0).cpu()
      input_    = inp.unsqueeze(0).to(device)
      restored = model_restoration(input_)
      restored = torch.clamp(restored[0],0,1)
      restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

      processed_image = img_as_ubyte(restored[0])
      processed_image = Image.fromarray(processed_image).resize(image.size)

    img_bytes = io.BytesIO()
    processed_image.save(img_bytes, format=image.format)
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type=f"image/{image.format.lower()}")
