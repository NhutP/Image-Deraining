import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output scaled between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# # Define the Convolutional Autoencoder
# import torch.nn.functional as F
# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()  # Output scaled between 0 and 1
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         # Resize output to match the target size (if necessary)
#         decoded_resized = F.interpolate(decoded, size=x.size()[2:], mode='bilinear', align_corners=False)
#         return decoded_resized




# # Custom Dataset for paired images (input and ground truth)
# class PairedImageDataset(Dataset):
#     def __init__(self, input_dir, groundtruth_dir, transform=None):
#         self.input_dir = input_dir
#         self.groundtruth_dir = groundtruth_dir
#         self.transform = transform
#         self.image_files = [f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         input_path = os.path.join(self.input_dir, self.image_files[idx])
#         groundtruth_path = os.path.join(self.groundtruth_dir, self.image_files[idx])

#         input_image = Image.open(input_path).convert('RGB')
#         groundtruth_image = Image.open(groundtruth_path).convert('RGB')

#         if self.transform:
#             input_image = self.transform(input_image)
#             groundtruth_image = self.transform(groundtruth_image)

#         return input_image, groundtruth_image

# class PairedImageDataset(Dataset):
#     def __init__(self, input_dir, groundtruth_dir, transform=None):
#         self.transform = transform
#         self.data = []
#         image_files = [f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
#         for img_file in image_files:
#             input_image = Image.open(os.path.join(input_dir, img_file)).convert('RGB')
#             groundtruth_image = Image.open(os.path.join(groundtruth_dir, img_file)).convert('RGB')
#             self.data.append((input_image, groundtruth_image))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         input_image, groundtruth_image = self.data[idx]

#         if self.transform:
#             input_image = self.transform(input_image)
#             groundtruth_image = self.transform(groundtruth_image)

#         return input_image, groundtruth_image

# Custom Dataset for paired images (input and ground truth)
class PairedImageDataset(Dataset):
    def __init__(self, input_dir, groundtruth_dir, transform=None, preload=False):
        self.transform = transform
        self.preload = preload
        self.data = []
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

        if preload:
            for img_file in self.image_files[:len(self.image_files)//2]:
                input_image = Image.open(os.path.join(input_dir, img_file)).convert('RGB')
                groundtruth_image = Image.open(os.path.join(groundtruth_dir, img_file)).convert('RGB')
                self.data.append((input_image, groundtruth_image))
        self.input_dir = input_dir
        self.groundtruth_dir = groundtruth_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.preload and idx < len(self.data):
            input_image, groundtruth_image = self.data[idx]
        else:
            img_file = self.image_files[idx]
            input_image = Image.open(os.path.join(self.input_dir, img_file)).convert('RGB')
            groundtruth_image = Image.open(os.path.join(self.groundtruth_dir, img_file)).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            groundtruth_image = self.transform(groundtruth_image)

        return input_image, groundtruth_image

# Dataset and DataLoader preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = PairedImageDataset(input_dir='./my_dataset/input', groundtruth_dir='./my_dataset/groundtruth')#, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

eval_dataset = PairedImageDataset(input_dir='./eval_dataset/input', groundtruth_dir='./eval_dataset/groundtruth')#, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Training the Autoencoder
def evaluate_model(model, eval_loader, device, criterion):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for inputs, groundtruths in eval_loader:
            inputs = inputs.to(device)
            groundtruths = groundtruths.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, groundtruths)
            eval_loss += loss.item()
    return eval_loss / len(eval_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 200
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for inputs, groundtruths in train_loader:
        inputs = inputs.to(device)
        groundtruths = groundtruths.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, groundtruths)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 5 == 0:
        eval_loss = evaluate_model(model, eval_loader, device, criterion)
        print(f"Epoch [{epoch}], Train Loss: {running_loss/len(train_loader):.4f}, Eval Loss: {eval_loss:.4f}")

        # Save model and evaluation information
        model_path = f"conv_autoencoder_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        with open("training_log.txt", "a") as log_file:
            log_file.write(f"Epoch: {epoch}, Train Loss: {running_loss/len(train_loader):.4f}, Eval Loss: {eval_loss:.4f}\n")

    if epoch % 2 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}")