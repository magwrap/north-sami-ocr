#!/usr/bin/env python3
"""Generate architecture diagram for CNN-CTC OCR pipeline."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

# Colors
colors = {
    'input': '#E8E8E8',
    'cnn': '#AED6F1',
    'encoder': '#ABEBC6',
    'output': '#F9E79F'
}

# Box parameters
box_width = 1.5
box_height = 1.8
y_center = 2

# Draw boxes
boxes = [
    (0.5, 'Input\nImage\n32×800', colors['input']),
    (2.5, '7-Layer\nCNN\nBackbone', colors['cnn']),
    (4.5, 'Adaptive\nPool\n(H→1)', colors['cnn']),
    (6.5, 'BiLSTM\nEncoder\n(2 layers)', colors['encoder']),
    (8.5, 'Linear +\nCTC\nDecoder', colors['encoder']),
    (10.5, 'Text\nOutput', colors['output']),
]

for x, label, color in boxes:
    rect = FancyBboxPatch(
        (x, y_center - box_height/2), box_width, box_height,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=color, edgecolor='black', linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + box_width/2, y_center, label,
            ha='center', va='center', fontsize=9, fontweight='bold')

# Draw arrows
arrow_style = "Simple,tail_width=0.5,head_width=4,head_length=6"
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + box_width
    x2 = boxes[i+1][0]
    arrow = FancyArrowPatch(
        (x1, y_center), (x2, y_center),
        arrowstyle=arrow_style,
        color='black'
    )
    ax.add_patch(arrow)

# Add dimension labels below
dims = [
    (0.5 + box_width/2, '(B,1,32,800)'),
    (2.5 + box_width/2, '(B,512,H\',W\')'),
    (4.5 + box_width/2, '(B,W\',512)'),
    (6.5 + box_width/2, '(B,W\',512)'),
    (8.5 + box_width/2, '(B,W\',396)'),
]

for x, label in dims:
    ax.text(x, y_center - box_height/2 - 0.3, label,
            ha='center', va='top', fontsize=7, style='italic', color='#555555')

# Title
ax.text(6, 3.7, 'CNN-CTC OCR Pipeline Architecture',
        ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('architecture.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('architecture.png', format='png', bbox_inches='tight', dpi=300)
print("Generated architecture.pdf and architecture.png")
