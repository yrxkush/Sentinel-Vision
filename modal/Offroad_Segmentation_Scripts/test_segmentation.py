"""
Segmentation Test Script
Runs inference on images and saves:
- outputs/mask.png (grayscale, class IDs 0-9)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse

# ============================================================================
# Configuration
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

# Class names for visualization
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Model: Enhanced Segmentation Head with UNet-style Decoder
# Must match training architecture exactly!
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and GELU activation."""
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
    """Spatial attention for feature refinement."""
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
    """Enhanced UNet-style decoder with attention and multi-scale features."""
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        # Initial projection from DINOv2 features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Decoder blocks with progressive upsampling
        self.decoder1 = ConvBlock(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock(256)

        self.decoder2 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock(128)

        self.decoder3 = ConvBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.decoder4 = ConvBlock(64, 32)

        # Final classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)

        # Stem
        x = self.stem(x)

        # Decoder with attention
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


# ============================================================================
# Main Function
# ============================================================================

def main():
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    parser = argparse.ArgumentParser(description='Segmentation inference script')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'segmentation_head.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(project_root, 'data', 'Offroad_Segmentation_testImages', 'Offroad_Segmentation_testImages'),
                        help='Path to test dataset')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(project_root, 'outputs'),
                        help='Directory to save outputs')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image dimensions (must match training)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    # Load DINOv2 backbone - MUST use BASE to match training!
    print("Loading DINOv2 backbone (base)...")
    BACKBONE_SIZE = "base"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension
    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # Load classifier - using UNet architecture to match training
    print(f"Loading model from {args.model_path}...")
    classifier = SegmentationHeadUNet(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")

    # Run inference on FIRST image only and save mask.png
    print("\nRunning inference on first image...")

    with torch.no_grad():
        # Get first image
        imgs, labels, data_ids = next(iter(val_loader))
        imgs = imgs.to(device)

        # Forward pass
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
        logits = classifier(output)
        outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

        # Get predicted mask (class IDs 0-9)
        predicted_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)

        # Save mask.png (grayscale, pixel values = class IDs)
        mask_path = os.path.join(args.output_dir, 'mask.png')
        mask_img = Image.fromarray(predicted_mask)
        mask_img.save(mask_path)
        print(f"Saved: {mask_path}")

        # Also save colored visualization for debugging
        color_mask = mask_to_color(predicted_mask)
        color_path = os.path.join(args.output_dir, 'mask_color.png')
        cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        print(f"Saved: {color_path}")

    print(f"\nInference complete!")
    print(f"Output files in {args.output_dir}/:")
    print(f"  - mask.png (grayscale, class IDs 0-9)")
    print(f"  - mask_color.png (colored visualization)")


if __name__ == "__main__":
    main()
