"""
SentinalVision - Off-Road Path Planner
Fixed version with proper point selection and improved heatmap
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

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
    def forward(self, x): 
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): 
        return x * self.attention(x)

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
# Configuration
# ============================================================================

COLOR_PALETTE = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235]
], dtype=np.uint8)

CLASS_NAMES = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

COST_MAP = {0: 1, 1: 3, 2: 4, 3: 2, 4: 5, 5: 6, 6: 15, 7: 50, 8: 1, 9: 0}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'modal', 'Offroad_Segmentation_Scripts', 'segmentation_head.pth')

W = int(((960 / 2) // 14) * 14)
H = int(((540 / 2) // 14) * 14)

DEVICE = None
BACKBONE = None
CLASSIFIER = None

# Global state
STATE = {
    'heatmap': None,
    'image': None,
    'start': None,
    'end': None
}

# ============================================================================
# Model Loading
# ============================================================================

def load_models():
    global DEVICE, BACKBONE, CLASSIFIER
    
    if BACKBONE is not None:
        return
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on {DEVICE}...")
    
    BACKBONE = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    BACKBONE.eval().to(DEVICE)
    
    CLASSIFIER = SegmentationHeadUNet(768, 10, W // 14, H // 14)
    CLASSIFIER.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    CLASSIFIER.eval().to(DEVICE)
    
    print("Models loaded!")

# ============================================================================
# Utility Functions
# ============================================================================

def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(10):
        color[mask == i] = COLOR_PALETTE[i]
    return color

def mask_to_heatmap(mask):
    heatmap = np.zeros_like(mask, dtype=np.float32)
    for cid, cost in COST_MAP.items():
        heatmap[mask == cid] = cost
    return heatmap

def create_heatmap_with_legend(heatmap):
    """Create heatmap visualization with legend (like previous version)."""
    import io
    
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    
    max_cost = max(COST_MAP.values())
    masked = np.ma.masked_where(heatmap == 0, heatmap)
    
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.LogNorm(vmin=1, vmax=max_cost)
    im = ax_main.imshow(masked, cmap=cmap, norm=norm)
    
    sky = (heatmap == 0).astype(float)
    ax_main.imshow(np.ma.masked_where(sky == 0, sky), 
                   cmap=mcolors.ListedColormap(['#87CEEB']), alpha=0.8)
    
    cbar = plt.colorbar(im, ax=ax_main, orientation='horizontal', shrink=0.8, pad=0.08)
    cbar.set_label('Traversal Cost (log scale)', fontsize=10, fontweight='bold')
    
    ax_main.set_title('Cost Heatmap', fontsize=12, fontweight='bold')
    ax_main.axis('off')
    
    # Legend
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.text(0.5, 0.95, 'Cost Legend', fontsize=12, fontweight='bold', ha='center', va='top')
    
    y_start, y_step = 0.85, 0.075
    for i, (cid, cost) in enumerate(COST_MAP.items()):
        y = y_start - i * y_step
        color = '#87CEEB' if cost == 0 else cmap(norm(max(cost, 1)))
        rect = plt.Rectangle((0.05, y - 0.025), 0.15, 0.05, facecolor=color, edgecolor='black')
        ax_legend.add_patch(rect)
        ax_legend.text(0.25, y, f"{CLASS_NAMES[cid]}: {cost}", fontsize=9, va='center')
    
    # Save to buffer and convert to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = np.array(Image.open(buf))[:, :, :3]  # Remove alpha channel if present
    plt.close(fig)
    buf.close()
    return img

def create_heatmap_no_legend(heatmap):
    """Create heatmap visualization WITHOUT legend - same colors as the legend version."""
    import io
    
    h, w = heatmap.shape
    max_cost = max(COST_MAP.values())
    
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    
    masked = np.ma.masked_where(heatmap == 0, heatmap)
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.LogNorm(vmin=1, vmax=max_cost)
    ax.imshow(masked, cmap=cmap, norm=norm)
    
    # Sky overlay
    sky = (heatmap == 0).astype(float)
    ax.imshow(np.ma.masked_where(sky == 0, sky), 
              cmap=mcolors.ListedColormap(['#87CEEB']), alpha=0.8)
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    img = np.array(Image.open(buf))[:, :, :3]
    plt.close(fig)
    buf.close()
    return img

def create_safe_overlay(image, heatmap):
    """Create safe area overlay with continuous gradient like heatmap."""
    h, w = heatmap.shape
    img = cv2.resize(image, (w, h)).astype(np.float32)
    overlay = img.copy()
    
    max_cost = 50.0
    
    # Apply continuous gradient for each pixel based on cost
    for cost_val in range(0, int(max_cost) + 1):
        mask = (heatmap >= cost_val) & (heatmap < cost_val + 1)
        if not np.any(mask):
            continue
            
        if cost_val == 0:
            # Sky - blue tint
            overlay[mask] = overlay[mask] * 0.3 + np.array([135, 206, 235]) * 0.7
        else:
            # Continuous gradient: Green -> Yellow -> Orange -> Red
            ratio = min(cost_val / max_cost, 1.0)
            
            if ratio < 0.1:  # Cost 1-5: Bright green
                color = np.array([0, 255, 50])
            elif ratio < 0.2:  # Cost 6-10: Yellow-green
                color = np.array([100, 230, 0])
            elif ratio < 0.3:  # Cost 11-15: Yellow
                color = np.array([200, 200, 0])
            elif ratio < 0.4:  # Cost 16-20: Yellow-orange
                color = np.array([255, 180, 0])
            elif ratio < 0.5:  # Cost 21-25: Orange
                color = np.array([255, 130, 0])
            elif ratio < 0.6:  # Cost 26-30: Dark orange
                color = np.array([255, 80, 0])
            elif ratio < 0.7:  # Cost 31-35: Light red
                color = np.array([255, 50, 0])
            elif ratio < 0.8:  # Cost 36-40: Red
                color = np.array([255, 20, 0])
            elif ratio < 0.9:  # Cost 41-45: Dark red
                color = np.array([220, 0, 0])
            else:  # Cost 46-50: Very dark red
                color = np.array([180, 0, 0])
            
            blend = 0.35 + (ratio * 0.25)  # More blend for higher danger
            overlay[mask] = overlay[mask] * (1 - blend) + color * blend
    
    return overlay.astype(np.uint8)

def create_simple_heatmap(heatmap):
    """Create simple heatmap without legend for path overlay."""
    h, w = heatmap.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Use RdYlGn colormap (green=safe, red=danger)
    max_cost = 50
    for cost_val in range(0, int(max_cost) + 1):
        mask = (heatmap >= cost_val) & (heatmap < cost_val + 1)
        if cost_val == 0:
            colored[mask] = [135, 206, 235]  # Sky blue
        else:
            # Normalize and map to green-yellow-red
            ratio = min(cost_val / max_cost, 1.0)
            if ratio < 0.5:
                # Green to Yellow
                r = int(255 * (ratio * 2))
                g = 255
            else:
                # Yellow to Red
                r = 255
                g = int(255 * (1 - (ratio - 0.5) * 2))
            colored[mask] = [r, g, 0]
    
    return colored

# ============================================================================
# A* Pathfinding
# ============================================================================

def astar(heatmap, start, end):
    h, w = heatmap.shape
    sr, sc = int(start[1]), int(start[0])
    er, ec = int(end[1]), int(end[0])
    
    sr, sc = max(0, min(h-1, sr)), max(0, min(w-1, sc))
    er, ec = max(0, min(h-1, er)), max(0, min(w-1, ec))
    
    pq = [(0, 0, sr, sc, [(sr, sc)])]
    visited = set()
    dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    def h_cost(a, b): return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    
    while pq:
        f, g, r, c, path = heappop(pq)
        if (r, c) == (er, ec): return path
        if (r, c) in visited: continue
        visited.add((r, c))
        
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                cost = heatmap[nr, nc]
                if cost == 0: cost = 500
                move = 1.414 if (dr != 0 and dc != 0) else 1.0
                new_g = g + cost * move
                heappush(pq, (new_g + h_cost((nr, nc), (er, ec)), new_g, nr, nc, path + [(nr, nc)]))
    return []

def draw_path_and_points(image, path, start, end):
    img = image.copy()
    
    # Draw path
    for i in range(len(path) - 1):
        pt1 = (int(path[i][1]), int(path[i][0]))
        pt2 = (int(path[i+1][1]), int(path[i+1][0]))
        cv2.line(img, pt1, pt2, (0, 255, 255), 3)
    
    # Draw start (green) and end (red)
    if start:
        cv2.circle(img, (int(start[0]), int(start[1])), 10, (0, 255, 0), -1)
        cv2.circle(img, (int(start[0]), int(start[1])), 10, (0, 0, 0), 2)
    if end:
        cv2.circle(img, (int(end[0]), int(end[1])), 10, (255, 0, 0), -1)
        cv2.circle(img, (int(end[0]), int(end[1])), 10, (0, 0, 0), 2)
    
    return img

# ============================================================================
# Main Functions
# ============================================================================

def run_segmentation(image):
    load_models()
    
    if image is None:
        return None, None, None, None, "❌ Please upload an image first!"
    
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    original = np.array(pil_image)
    
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = BACKBONE.forward_features(tensor)["x_norm_patchtokens"]
        logits = CLASSIFIER(features)
        outputs = F.interpolate(logits, size=tensor.shape[2:], mode="bilinear", align_corners=False)
        mask = torch.argmax(outputs, dim=1)[0].cpu().numpy().astype(np.uint8)
    
    mask_color = mask_to_color(mask)
    heatmap = mask_to_heatmap(mask)
    heatmap_visual = create_heatmap_with_legend(heatmap)
    safe_overlay = create_safe_overlay(original, heatmap)
    heatmap_no_legend = create_heatmap_no_legend(heatmap)  # Create heatmap without legend for path output
    
    # Save to state (store all visuals for path overlay)
    STATE['heatmap'] = heatmap
    STATE['image'] = cv2.resize(original, (heatmap.shape[1], heatmap.shape[0]))
    STATE['safe_overlay'] = safe_overlay
    STATE['heatmap_no_legend'] = heatmap_no_legend  # Store the heatmap WITHOUT legend
    STATE['start'] = None
    STATE['end'] = None
    
    return mask_color, heatmap_visual, safe_overlay, safe_overlay.copy(), "✅ Ready! Enter start/end coordinates below."

def update_preview(start_x, start_y, end_x, end_y):
    """Live preview of points as sliders move (no path yet)."""
    if STATE['heatmap'] is None or STATE['image'] is None:
        return None
    
    h, w = STATE['heatmap'].shape
    sx = int(start_x * w / 100)
    sy = int(start_y * h / 100)
    ex = int(end_x * w / 100)
    ey = int(end_y * h / 100)
    
    # Draw just the points
    img = STATE['image'].copy()
    cv2.circle(img, (sx, sy), 12, (0, 255, 0), -1)  # Green start
    cv2.circle(img, (sx, sy), 12, (0, 0, 0), 2)
    cv2.circle(img, (ex, ey), 12, (255, 0, 0), -1)  # Red end
    cv2.circle(img, (ex, ey), 12, (0, 0, 0), 2)
    
    # Draw line between points
    cv2.line(img, (sx, sy), (ex, ey), (255, 255, 0), 1, cv2.LINE_AA)
    
    return img

def update_points_and_path(start_x, start_y, end_x, end_y):
    if STATE['heatmap'] is None or STATE['image'] is None:
        return None, "❌ Run segmentation first!"
    
    h, w = STATE['heatmap'].shape
    
    # Scale coordinates to heatmap size (assuming input is 0-100 percentage)
    sx = int(start_x * w / 100)
    sy = int(start_y * h / 100)
    ex = int(end_x * w / 100)
    ey = int(end_y * h / 100)
    
    STATE['start'] = (sx, sy)
    STATE['end'] = (ex, ey)
    
    # Compute path
    path = astar(STATE['heatmap'], STATE['start'], STATE['end'])
    
    if not path:
        result = draw_path_and_points(STATE['image'], [], STATE['start'], STATE['end'])
        return result, "❌ No path found! Try different points."
    
    # Draw path on all three views (using prebuilt visuals from STATE)
    # 1. Original image
    img_original = draw_path_and_points(STATE['image'].copy(), path, STATE['start'], STATE['end'])
    
    # 2. Safe overlay (use prebuilt)
    img_safe = draw_path_and_points(STATE['safe_overlay'].copy(), path, STATE['start'], STATE['end'])
    
    # 3. Heatmap - use the prebuilt heatmap WITHOUT legend (same colors, no overlap)
    heatmap_resized = cv2.resize(STATE['heatmap_no_legend'], (w, h))
    img_heatmap = draw_path_and_points(heatmap_resized, path, STATE['start'], STATE['end'])
    
    # Combine into one image (3 columns)
    combined = np.hstack([img_original, img_safe, img_heatmap])
    
    # Add labels with shadow for visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (12, 27), font, 0.7, (0, 0, 0), 3)
    cv2.putText(combined, "Original", (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Safe Areas", (w + 12, 27), font, 0.7, (0, 0, 0), 3)
    cv2.putText(combined, "Safe Areas", (w + 10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Heatmap", (2*w + 12, 27), font, 0.7, (0, 0, 0), 3)
    cv2.putText(combined, "Heatmap", (2*w + 10, 25), font, 0.7, (255, 255, 255), 2)
    
    total_cost = sum(STATE['heatmap'][p[0], p[1]] for p in path)
    
    return combined, f"✅ Path found! {len(path)} steps, Cost: {total_cost:.0f}"

# ============================================================================
# Gradio Interface
# ============================================================================

def create_ui():
    with gr.Blocks(title="SentinalVision") as demo:
        gr.Markdown("""
        # 🛰️ SentinalVision - Off-Road Path Planner
        **Upload image → Run segmentation → Set coordinates → Find safest path**
        """)
        
        # Section 1: Upload and Segment
        gr.Markdown("## 📷 Step 1: Upload & Analyze")
        with gr.Row():
            input_image = gr.Image(label="Upload Image", type="numpy")
            segment_btn = gr.Button("🔍 Run Segmentation", variant="primary", size="lg")
        
        status = gr.Textbox(label="Status", interactive=False)
        
        # Section 2: Results
        gr.Markdown("## 🎨 Step 2: View Results")
        with gr.Row():
            seg_output = gr.Image(label="Segmentation Mask")
            heatmap_output = gr.Image(label="Cost Heatmap with Legend")
        
        with gr.Row():
            safe_output = gr.Image(label="Safe Areas (Green=Safe, Yellow=Caution, Red=Danger)")
        
        # Section 3: Path Planning
        gr.Markdown("## 🛤️ Step 3: Plan Path")
        gr.Markdown("**Move sliders to position points - preview updates live!**")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**🟢 Start Point (Green)**")
                start_x = gr.Slider(0, 100, value=50, label="X position (%)", interactive=True)
                start_y = gr.Slider(0, 100, value=90, label="Y position (%)", interactive=True)
            with gr.Column(scale=1):
                gr.Markdown("**🔴 End Point (Red)**")
                end_x = gr.Slider(0, 100, value=50, label="X position (%)", interactive=True)
                end_y = gr.Slider(0, 100, value=10, label="Y position (%)", interactive=True)
        
        with gr.Row():
            preview_output = gr.Image(label="📍 Live Preview - Adjust sliders to move points")
        
        path_btn = gr.Button("🚀 Find Safest Path", variant="primary", size="lg")
        path_status = gr.Textbox(label="Path Status", interactive=False)
        
        with gr.Row():
            path_output = gr.Image(label="🗺️ Computed Path (Cyan line)")
        
        # Cost reference
        gr.Markdown("""
        ---
        ### 📊 Traversal Cost Reference
        | Class | Cost | | Class | Cost |
        |-------|------|-|-------|------|
        | Background | 1 | | Ground Clutter | 6 |
        | Trees | 3 | | Logs | 15 |
        | Lush Bushes | 4 | | **Rocks** | **50** |
        | Dry Grass | 2 | | Landscape | 1 |
        | Dry Bushes | 5 | | Sky | 0 (impassable) |
        """)
        
        # Event handlers
        segment_btn.click(
            fn=run_segmentation,
            inputs=[input_image],
            outputs=[seg_output, heatmap_output, safe_output, preview_output, status]
        )
        
        # Live preview on slider change
        for slider in [start_x, start_y, end_x, end_y]:
            slider.change(
                fn=update_preview,
                inputs=[start_x, start_y, end_x, end_y],
                outputs=[preview_output]
            )
        
        path_btn.click(
            fn=update_points_and_path,
            inputs=[start_x, start_y, end_x, end_y],
            outputs=[path_output, path_status]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting SentinalVision...")
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)

