"""
Model Stats Comparison: 10 Epochs vs 50 Epochs
Creates a multi-panel visualization comparing all metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Stats from training results
stats = {
    'metrics': ['Val IoU', 'Dice Score', 'Pixel Accuracy', 'Val Loss', 'Train Loss'],
    '10_epochs': [0.2958, 0.4416, 0.7036, 0.8148, 0.8203],  # Original model
    '50_epochs': [0.5046, 0.7026, 0.8125, 0.3605, 0.3040],  # Improved model
}

# Calculate improvements
improvements = []
for i, metric in enumerate(stats['metrics']):
    old = stats['10_epochs'][i]
    new = stats['50_epochs'][i]
    if 'Loss' in metric:
        pct = ((old - new) / old) * 100  # Lower is better
    else:
        pct = ((new - old) / old) * 100  # Higher is better
    improvements.append(pct)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Model Performance: 10 Epochs vs 50 Epochs', fontsize=18, fontweight='bold', y=0.98)

# Color scheme
colors_10 = '#FF6B6B'  # Red for old
colors_50 = '#4ECDC4'  # Teal for new

# 1. Bar chart comparison
ax1 = fig.add_subplot(2, 3, 1)
x = np.arange(len(stats['metrics']))
width = 0.35
bars1 = ax1.bar(x - width/2, stats['10_epochs'], width, label='10 Epochs', color=colors_10, edgecolor='black')
bars2 = ax1.bar(x + width/2, stats['50_epochs'], width, label='50 Epochs', color=colors_50, edgecolor='black')
ax1.set_ylabel('Value', fontsize=11)
ax1.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(stats['metrics'], rotation=45, ha='right', fontsize=9)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# 2. IoU Comparison (dedicated)
ax2 = fig.add_subplot(2, 3, 2)
bars = ax2.bar(['10 Epochs', '50 Epochs'], [stats['10_epochs'][0], stats['50_epochs'][0]], 
               color=[colors_10, colors_50], edgecolor='black', linewidth=2)
ax2.set_ylabel('IoU Score', fontsize=11)
ax2.set_title('Mean IoU (mIoU)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 0.7)
for bar, val in zip(bars, [stats['10_epochs'][0], stats['50_epochs'][0]]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target: 0.5')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Dice Score Comparison
ax3 = fig.add_subplot(2, 3, 3)
bars = ax3.bar(['10 Epochs', '50 Epochs'], [stats['10_epochs'][1], stats['50_epochs'][1]], 
               color=[colors_10, colors_50], edgecolor='black', linewidth=2)
ax3.set_ylabel('Dice Score', fontsize=11)
ax3.set_title('Dice Score (F1)', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.0)
for bar, val in zip(bars, [stats['10_epochs'][1], stats['50_epochs'][1]]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# 4. Pixel Accuracy
ax4 = fig.add_subplot(2, 3, 4)
bars = ax4.bar(['10 Epochs', '50 Epochs'], [stats['10_epochs'][2]*100, stats['50_epochs'][2]*100], 
               color=[colors_10, colors_50], edgecolor='black', linewidth=2)
ax4.set_ylabel('Accuracy (%)', fontsize=11)
ax4.set_title('Pixel Accuracy', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 100)
for bar, val in zip(bars, [stats['10_epochs'][2]*100, stats['50_epochs'][2]*100]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
ax4.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target: 80%')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Loss Comparison
ax5 = fig.add_subplot(2, 3, 5)
x = np.arange(2)
width = 0.35
bars1 = ax5.bar(x - width/2, [stats['10_epochs'][3], stats['50_epochs'][3]], width, 
                label='Val Loss', color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
bars2 = ax5.bar(x + width/2, [stats['10_epochs'][4], stats['50_epochs'][4]], width, 
                label='Train Loss', color=['#FFB6B6', '#A8E6CF'], edgecolor='black')
ax5.set_ylabel('Loss', fontsize=11)
ax5.set_title('Loss (Lower is Better)', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(['10 Epochs', '50 Epochs'])
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Improvement Percentages
ax6 = fig.add_subplot(2, 3, 6)
colors = ['#4ECDC4' if imp > 0 else '#FF6B6B' for imp in improvements]
bars = ax6.barh(stats['metrics'], improvements, color=colors, edgecolor='black')
ax6.set_xlabel('Improvement (%)', fontsize=11)
ax6.set_title('% Improvement (50 vs 10 Epochs)', fontsize=12, fontweight='bold')
ax6.axvline(x=0, color='black', linewidth=1)
for bar, val in zip(bars, improvements):
    x_pos = bar.get_width() + 2 if val > 0 else bar.get_width() - 2
    ha = 'left' if val > 0 else 'right'
    ax6.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%', 
             ha=ha, va='center', fontweight='bold', fontsize=10)
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Add summary text at bottom
summary = (
    f"Summary: IoU improved from {stats['10_epochs'][0]:.3f} → {stats['50_epochs'][0]:.3f} (+{improvements[0]:.1f}%)  |  "
    f"Accuracy: {stats['10_epochs'][2]*100:.1f}% → {stats['50_epochs'][2]*100:.1f}%  |  "
    f"Val Loss decreased by {-improvements[3]:.1f}%"
)
fig.text(0.5, 0.01, summary, ha='center', fontsize=11, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_path = os.path.join(project_root, 'outputs', 'model_stats_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
print("\nModel Comparison Summary:")
print("=" * 50)
for i, metric in enumerate(stats['metrics']):
    print(f"{metric:15s}: {stats['10_epochs'][i]:.4f} → {stats['50_epochs'][i]:.4f} ({improvements[i]:+.1f}%)")
