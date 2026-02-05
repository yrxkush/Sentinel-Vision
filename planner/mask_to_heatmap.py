"""
Mask to Heatmap Converter
Converts segmentation mask (class IDs) to cost heatmap for path planning.

Input:  outputs/mask.png (grayscale, pixel values = class IDs 0-9)
Output: outputs/heatmap.npy (numpy array with traversal costs)
        outputs/heatmap_vis.png (colored visualization)
"""

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================================
# Cost Map Configuration
# ============================================================================

COST_MAP = {
    0: 1,     # Background - easy (open ground)
    1: 3,     # Trees - moderate obstacle
    2: 4,     # Lush Bushes - moderate obstacle
    3: 2,     # Dry Grass - easy
    4: 5,     # Dry Bushes - harder
    5: 6,     # Ground Clutter - harder
    6: 15,    # Logs - significant obstacle
    7: 50,    # Rocks - very high cost (nearly impassable)
    8: 1,     # Landscape - easy (open terrain)
    9: 0,     # Sky - ignored (not traversable ground)
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]


def mask_to_heatmap(mask: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to cost heatmap."""
    heatmap = np.zeros_like(mask, dtype=np.float32)
    for class_id, cost in COST_MAP.items():
        heatmap[mask == class_id] = cost
    return heatmap


def create_heatmap_visualization(heatmap: np.ndarray, output_path: str):
    """Create colored visualization of the heatmap with legend."""
    # Create figure with GridSpec for proper layout
    fig = plt.figure(figsize=(16, 8))
    
    # Main heatmap takes 75% width, legend takes 25%
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    
    # Use log scale for better distinction between low and high costs
    max_cost = max(COST_MAP.values())
    masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
    
    # Use a more distinguishable colormap with log normalization
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.LogNorm(vmin=1, vmax=max_cost)
    im = ax_main.imshow(masked_heatmap, cmap=cmap, norm=norm)
    
    # Show sky (cost=0) in blue
    sky_mask = (heatmap == 0).astype(float)
    ax_main.imshow(np.ma.masked_where(sky_mask == 0, sky_mask), 
              cmap=mcolors.ListedColormap(['#87CEEB']), alpha=0.8)
    
    # Colorbar below the main image
    cbar = plt.colorbar(im, ax=ax_main, orientation='horizontal', 
                        shrink=0.8, pad=0.08, aspect=30)
    cbar.set_label('Traversal Cost (log scale)', fontsize=11, fontweight='bold')
    
    ax_main.set_title('Heatmap: Traversal Cost Map', fontsize=16, fontweight='bold', pad=10)
    ax_main.axis('off')
    
    # Create beautiful legend in the right panel
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    # Title for legend
    ax_legend.text(0.5, 0.95, 'Cost Legend', fontsize=14, fontweight='bold',
                   ha='center', va='top', transform=ax_legend.transAxes)
    
    # Draw color boxes with class names
    y_start = 0.85
    y_step = 0.075
    box_width = 0.15
    box_height = 0.05
    
    for i, (class_id, cost) in enumerate(COST_MAP.items()):
        y_pos = y_start - i * y_step
        
        # Get color for this cost (use colormap or special color for sky)
        if cost == 0:
            color = '#87CEEB'  # Sky blue
        else:
            color = cmap(norm(cost))
        
        # Draw color box
        rect = plt.Rectangle((0.05, y_pos - box_height/2), box_width, box_height,
                             facecolor=color, edgecolor='black', linewidth=1,
                             transform=ax_legend.transAxes, clip_on=False)
        ax_legend.add_patch(rect)
        
        # Add class name and cost
        label = f"{CLASS_NAMES[class_id]}: {cost}"
        ax_legend.text(0.25, y_pos, label, fontsize=10, va='center', 
                       fontfamily='sans-serif', transform=ax_legend.transAxes)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_root, 'outputs')
    
    mask_path = os.path.join(outputs_dir, 'mask.png')
    heatmap_npy_path = os.path.join(outputs_dir, 'heatmap.npy')
    heatmap_vis_path = os.path.join(outputs_dir, 'heatmap_vis.png')
    
    if not os.path.exists(mask_path):
        print(f"ERROR: mask.png not found at {mask_path}")
        print("Please run test_segmentation.py first.")
        return
    
    print(f"Loading mask from: {mask_path}")
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img, dtype=np.uint8)
    print(f"Mask shape: {mask.shape}")
    print(f"Unique class IDs: {np.unique(mask)}")
    
    print("\nConverting mask to heatmap...")
    heatmap = mask_to_heatmap(mask)
    print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}, mean: {heatmap.mean():.2f}")
    
    np.save(heatmap_npy_path, heatmap)
    print(f"\nSaved: {heatmap_npy_path}")
    
    create_heatmap_visualization(heatmap, heatmap_vis_path)
    print(f"Saved: {heatmap_vis_path}")
    
    print("\n" + "=" * 50)
    print("HEATMAP CONVERSION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
