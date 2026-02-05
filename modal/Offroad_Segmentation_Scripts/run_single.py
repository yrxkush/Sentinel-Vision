"""
Quick inference script for a single image.
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

# Model architecture (must match training)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attention(x)

class SegmentationHeadUNet(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )
        self.decoder1 = ConvBlock(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock(256)
        self.decoder2 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock(128)
        self.decoder3 = ConvBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(64, 32)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.decoder1(x)
        x = self.up1(x)
        x = self.attn1(x)
        x = self.decoder2(x)
        x = self.up2(x)
        x = self.attn2(x)
        x = self.decoder3(x)
        x = self.up3(x)
        x = self.decoder4(x)
        return self.classifier(x)

# Color palette
color_palette = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235],
], dtype=np.uint8)

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(10):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
image_path = os.path.join(project_root, 'data', 'Offroad_Segmentation_Training_Dataset', 
                          'Offroad_Segmentation_Training_Dataset', 'train', 'Color_Images', 'cc0000012.png')
model_path = os.path.join(script_dir, 'segmentation_head.pth')
output_dir = os.path.join(project_root, 'outputs')

print(f"Image: {image_path}")
print(f"Model: {model_path}")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image
image = Image.open(image_path).convert("RGB")
img_tensor = transform(image).unsqueeze(0).to(device)

# Load backbone
print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
backbone.eval().to(device)

# Load classifier
classifier = SegmentationHeadUNet(768, 10, w // 14, h // 14)
classifier.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
classifier.eval().to(device)
print("Model loaded!")

# Inference
print("Running inference...")
with torch.no_grad():
    features = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
    logits = classifier(features)
    outputs = F.interpolate(logits, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
    pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)

# Save outputs
os.makedirs(output_dir, exist_ok=True)
mask_path = os.path.join(output_dir, 'mask.png')
color_path = os.path.join(output_dir, 'mask_color.png')

Image.fromarray(pred_mask).save(mask_path)
cv2.imwrite(color_path, cv2.cvtColor(mask_to_color(pred_mask), cv2.COLOR_RGB2BGR))

print(f"\nSaved: {mask_path}")
print(f"Saved: {color_path}")
print("\nDone!")
