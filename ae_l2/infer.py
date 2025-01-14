import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder: Downsample input image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Input channels: 3 (RGB), Output: 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample, Output: 128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample, Output: 256
            nn.ReLU(inplace=True),
        )

        # Bottleneck: Further processing at the latent space
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder: Upsample to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample, Output: 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample, Output: 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output channels: 3 (RGB)
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Forward pass through encoder, bottleneck, and decoder
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# Load checkpoint and model
def load_model(checkpoint_path):
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Inference on a single image
def infer_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((480, 720)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu()  # Remove batch dimension

    output_image = transforms.ToPILImage()(output_tensor)
    return output_image

# Process a folder of images
def process_folder(model, input_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_path):
            output_image = infer_image(model, input_path, device)
            output_image.save(output_path)
        elif os.path.isdir(input_path):
            process_folder(model, input_path, output_path, device)  # Recursive for subfolders

# Main function for inference
def main(checkpoint_path, test_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path).to(device)

    for subfolder in os.listdir(test_folder):
        input_subfolder = os.path.join(test_folder, subfolder, "input")
        output_subfolder = os.path.join(output_folder, subfolder)

        if os.path.isdir(input_subfolder):
            process_folder(model, input_subfolder, output_subfolder, device)

if __name__ == "__main__":
    checkpoint_path = "./conv_autoencoder_epoch_140.pth"  # Path to the .pth file
    test_folder = "./Datasets/test/"             # Path to the test folder containing subfolders
    output_folder = "./results"    # Path to the output folder

    main(checkpoint_path, test_folder, output_folder)
