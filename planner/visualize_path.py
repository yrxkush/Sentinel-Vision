"""
Safe Area Visualization and Comparison Generator
Highlights safe traversable areas on the original image.
"""

import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

def create_safe_area_overlay(original, heatmap, safe_threshold=3):
    """
    Create an overlay highlighting safe areas (low cost regions).
    Safe areas = cost <= threshold (Background, Dry Grass, Landscape)
    """
    # Resize original to match heatmap
    img = cv2.resize(original, (heatmap.shape[1], heatmap.shape[0]))
    
    # Create mask for safe areas (low cost, excluding sky which is 0)
    safe_mask = (heatmap > 0) & (heatmap <= safe_threshold)
    
    # Create colored overlay
    overlay = img.copy()
    
    # Green tint for safe areas
    overlay[safe_mask] = overlay[safe_mask] * 0.5 + np.array([0, 180, 0]) * 0.5
    
    # Red tint for dangerous areas (high cost)
    danger_mask = heatmap >= 15  # Logs and Rocks
    overlay[danger_mask] = overlay[danger_mask] * 0.5 + np.array([0, 0, 200]) * 0.5
    
    # Yellow for medium risk
    medium_mask = (heatmap > safe_threshold) & (heatmap < 15)
    overlay[medium_mask] = overlay[medium_mask] * 0.7 + np.array([0, 180, 255]) * 0.3
    
    return overlay.astype(np.uint8)


def create_comparison_image(original_path, mask_color_path, heatmap_vis_path, safe_area_img, output_path):
    """Create a 2x2 comparison grid of all outputs."""
    # Load images
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    mask_color = cv2.imread(mask_color_path)
    mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    
    heatmap_vis = cv2.imread(heatmap_vis_path)
    heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
    
    safe_rgb = cv2.cvtColor(safe_area_img, cv2.COLOR_BGR2RGB)
    
    # Resize all to same size
    target_h, target_w = 270, 480
    original = cv2.resize(original, (target_w, target_h))
    mask_color = cv2.resize(mask_color, (target_w, target_h))
    safe_rgb = cv2.resize(safe_rgb, (target_w, target_h))
    heatmap_vis = cv2.resize(heatmap_vis, (target_w, target_h))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_color)
    axes[0, 1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(heatmap_vis)
    axes[1, 0].set_title('Traversal Cost Heatmap', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(safe_rgb)
    axes[1, 1].set_title('Safe Areas (Green=Safe, Yellow=Caution, Red=Danger)', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('SentinalVision: Off-Road Segmentation Pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, 'outputs')
    
    original_path = os.path.join(project_root, 'data', 'Offroad_Segmentation_Training_Dataset',
                                 'Offroad_Segmentation_Training_Dataset', 'train', 'Color_Images', 'cc0000012.png')
    mask_color_path = os.path.join(outputs_dir, 'mask_color.png')
    heatmap_npy_path = os.path.join(outputs_dir, 'heatmap.npy')
    heatmap_vis_path = os.path.join(outputs_dir, 'heatmap_vis.png')
    
    print("Loading data...")
    heatmap = np.load(heatmap_npy_path)
    original = cv2.imread(original_path)
    print(f"Heatmap shape: {heatmap.shape}")
    
    print("Creating safe area overlay...")
    safe_area_img = create_safe_area_overlay(original, heatmap, safe_threshold=3)
    
    # Save safe area overlay
    safe_output = os.path.join(outputs_dir, 'safe_areas.png')
    cv2.imwrite(safe_output, safe_area_img)
    print(f"Saved: {safe_output}")
    
    # Create comparison
    print("Creating comparison image...")
    comparison_output = os.path.join(outputs_dir, 'comparison.png')
    create_comparison_image(original_path, mask_color_path, heatmap_vis_path, safe_area_img, comparison_output)
    print(f"Saved: {comparison_output}")
    
    print("\n" + "=" * 50)
    print("SAFE AREA VISUALIZATION COMPLETE")
    print("=" * 50)
    print("\nColor coding:")
    print("  GREEN  = Safe (cost 1-3: Background, Dry Grass, Landscape)")
    print("  YELLOW = Caution (cost 4-14: Bushes, Ground Clutter)")
    print("  RED    = Danger (cost 15+: Logs, Rocks)")
    print("  BLUE   = Sky (ignored)")


if __name__ == "__main__":
    main()
