"""
Batch Pipeline: Run full segmentation + heatmap + visualization on multiple images.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================================
# Model Architecture
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.attention(x)

class SegmentationHeadUNet(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 512, 1), nn.BatchNorm2d(512), nn.GELU())
        self.decoder1 = ConvBlock(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.attn1 = AttentionBlock(256)
        self.decoder2 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.attn2 = AttentionBlock(128)
        self.decoder3 = ConvBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder4 = ConvBlock(64, 32)
        self.classifier = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(32, out_channels, 1))

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.attn1(self.up1(self.decoder1(x)))
        x = self.attn2(self.up2(self.decoder2(x)))
        x = self.up3(self.decoder3(x))
        x = self.decoder4(x)
        return self.classifier(x)

# ============================================================================
# Utilities
# ============================================================================

color_palette = np.array([
    [0,0,0], [34,139,34], [0,255,0], [210,180,140], [139,90,43],
    [128,128,0], [139,69,19], [128,128,128], [160,82,45], [135,206,235]
], dtype=np.uint8)

COST_MAP = {0:1, 1:3, 2:4, 3:2, 4:5, 5:6, 6:15, 7:50, 8:1, 9:0}

def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(10): color[mask == i] = color_palette[i]
    return color

def mask_to_heatmap(mask):
    heatmap = np.zeros_like(mask, dtype=np.float32)
    for cid, cost in COST_MAP.items(): heatmap[mask == cid] = cost
    return heatmap

def create_safe_overlay(original, heatmap):
    img = cv2.resize(original, (heatmap.shape[1], heatmap.shape[0]))
    overlay = img.copy().astype(np.float32)
    safe = (heatmap > 0) & (heatmap <= 3)
    danger = heatmap >= 15
    caution = (heatmap > 3) & (heatmap < 15)
    overlay[safe] = overlay[safe] * 0.5 + np.array([0, 180, 0]) * 0.5
    overlay[danger] = overlay[danger] * 0.5 + np.array([0, 0, 200]) * 0.5
    overlay[caution] = overlay[caution] * 0.7 + np.array([0, 180, 255]) * 0.3
    return overlay.astype(np.uint8)

def create_comparison(original, mask_color, heatmap, safe_overlay, output_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    h, w = heatmap.shape
    orig_resized = cv2.resize(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), (w, h))
    
    axes[0,0].imshow(orig_resized)
    axes[0,0].set_title('Original', fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('Segmentation', fontweight='bold')
    axes[0,1].axis('off')
    
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.LogNorm(vmin=1, vmax=50)
    masked = np.ma.masked_where(heatmap == 0, heatmap)
    axes[1,0].imshow(masked, cmap=cmap, norm=norm)
    axes[1,0].imshow(np.ma.masked_where(heatmap != 0, np.ones_like(heatmap)), 
                     cmap=mcolors.ListedColormap(['#87CEEB']), alpha=0.8)
    axes[1,0].set_title('Heatmap', fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(cv2.cvtColor(safe_overlay, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title('Safe Areas (G=Safe, Y=Caution, R=Danger)', fontsize=9, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.suptitle(f'SentinalVision: {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, 'modal', 'Offroad_Segmentation_Scripts', 'segmentation_head.pth')
    images_dir = os.path.join(project_root, 'data', 'Offroad_Segmentation_Training_Dataset',
                              'Offroad_Segmentation_Training_Dataset', 'train', 'Color_Images')
    output_dir = os.path.join(project_root, 'outputs', 'batch_comparisons')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get random images
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    random.seed(42)
    selected = random.sample(all_images, 5)
    print(f"Selected images: {selected}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    w, h = int(((960/2)//14)*14), int(((540/2)//14)*14)
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load models
    print("Loading backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval().to(device)
    
    print("Loading classifier...")
    classifier = SegmentationHeadUNet(768, 10, w//14, h//14)
    classifier.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    classifier.eval().to(device)
    
    # Process each image
    for i, img_name in enumerate(selected, 1):
        print(f"\n[{i}/5] Processing {img_name}...")
        img_path = os.path.join(images_dir, img_name)
        
        # Load and inference
        original = cv2.imread(img_path)
        pil_img = Image.open(img_path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = backbone.forward_features(tensor)["x_norm_patchtokens"]
            logits = classifier(features)
            outputs = F.interpolate(logits, size=tensor.shape[2:], mode="bilinear", align_corners=False)
            pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # Generate all outputs
        mask_color = mask_to_color(pred_mask)
        heatmap = mask_to_heatmap(pred_mask)
        safe_overlay = create_safe_overlay(original, heatmap)
        
        # Save comparison
        base = os.path.splitext(img_name)[0]
        output_path = os.path.join(output_dir, f'{base}_comparison.png')
        create_comparison(original, mask_color, heatmap, safe_overlay, output_path, img_name)
        print(f"Saved: {output_path}")
    
    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"\n5 comparison images saved to: {output_dir}/")

if __name__ == "__main__":
    main()
