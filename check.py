import os
from PIL import Image
import matplotlib.pyplot as plt
import random

# Paths
img_dir = 'data/TinyAgri/Tomatoes/scene1'
mask_dir = 'data/masks/Tomatoes/scene1'

# Get sorted list of image files
img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png'))])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png'))])

# Ensure reproducibility (optional: set a seed)
random.seed(42)

# Take first 50 images
num_samples = min(50, len(img_files))
img_files = img_files[:num_samples]

plt.figure(figsize=(20, 4 * num_samples))

for idx, img_filename in enumerate(img_files):
    img_path = os.path.join(img_dir, img_filename)

    # Get corresponding mask (maskX.png where X is position number)
    mask_filename = f"mask{idx + 1}.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    
    # Load mask if it exists, otherwise create blank mask
    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        mask_title = f'Mask: {mask_filename}'
    else:
        mask = Image.new('L', Image.open(img_path).size, 0)
        mask_title = 'Mask: MISSING'

    img = Image.open(img_path)

    plt.subplot(num_samples, 2, 2 * idx + 1)
    plt.imshow(img)
    plt.title(f'Image: {img_filename}')
    plt.axis('off')

    plt.subplot(num_samples, 2, 2 * idx + 2)
    plt.imshow(mask, cmap='gray')
    plt.title(mask_title)
    plt.axis('off')

plt.tight_layout()
plt.savefig('image_mask_grid.png')  # Save the figure to a file
plt.close()