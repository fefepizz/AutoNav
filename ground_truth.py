import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pandas as pd
import os
import subprocess

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def show_anns(anns):
    if len(anns) == 0:
        return
    # orders by area
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # creates a template blank image where data and mask will be overlayed
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
      m = ann['segmentation']
      color_mask = np.concatenate([np.random.random(3), [0.35]])
      img[m] = color_mask
    ax.imshow(img)
    
def largest_ann(anns):
    if len(anns) == 0:
        return
    # orders by area
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ann = sorted_anns[0]

    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = ann['segmentation']

    # Applying intensity scaling to the mask
    img = img.astype(np.float32)  # Convert to float32 for scaling
    img = (img - img.min()) / (img.max() - img.min())

    img = img * 0.3  # Scale intensity to be below 0.35

    ax.imshow(img, cmap='gray') # Display using grayscale colormap

def load_images(images_path, start_idx=20, end_idx=21):
    image_files = sorted(os.listdir(images_path))[start_idx:end_idx]
    images = []
    
    for img_file in image_files:
        image_full_path = os.path.join(images_path, img_file)
        img = cv2.imread(image_full_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
    
    return images

# Use os.path.join for data directory
crops_dir = os.path.join("data", "TinyAgri", "Crops", "scene2")
images = load_images(crops_dir, start_idx=0, end_idx=len(os.listdir(crops_dir)))


model_path = os.path.join("models", "sam_vit_h_4b8939.pth")
os.makedirs(os.path.dirname(model_path), exist_ok=True)

if not os.path.isfile(model_path):
    print("SAM model not found, downloading...")
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    subprocess.run(["wget", "-O", model_path, model_url], check=True)
    print("Download complete.")

sam_model = model_path
model_type = "vit_h"

device = "cpu"
if torch.cuda.is_available():
  device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_model)
sam.to(device= device)

mask_gen = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=4,  # Lower for coarser masks, helps merge small regions, 8 before
    pred_iou_thresh=0.7,  # Lower to merge more regions
    stability_score_thresh=0.7,  # Lower to allow less stable (but larger) masks
    crop_n_layers=1,  # Default, for speed
    crop_n_points_downscale_factor=2,  # Default
    min_mask_region_area=5000,  # Much higher to ignore small plants/leaves
)

masks = []
for i, img in enumerate(images):
  anns = mask_gen.generate(img)
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ann = sorted_anns[0]
  masks.append(ann['segmentation'].astype(np.uint8))
  print("Done with mask: ", i)

line_points = np.zeros((len(masks), 11), dtype=object)

for l, m in enumerate(masks):

    height = len(m)
    width = len(m[0])
    last_p = np.zeros(width, dtype=int)

    # Process each column
    for x in range(width):

        # Start from top and move down until we find a non-zero pixel
        y = 0
        while y < height - 1:
            if m[y][x] == 1:
                last_p[x] = y
                break
            y += 1

    # Find minimum non-zero y-position
    min_val = height
    for val in last_p:
        if val > 0 and val < min_val:
            min_val = val
            
    # we can assume the vanishing point is not too close to the border
    # and that the mask covers most of the picture
    if 30 < min_val < height and np.mean(m) > 0.4:
        # find all columns with minimum y-position
        indices = [i for i, val in enumerate(last_p) if val == min_val]
        median_x = indices[len(indices) // 2] # median
        v_point = (int(median_x), int(min_val))
        print(f"Mask {l}, v_point: {v_point}")

        # Sample points at regular intervals
        line_points[l, 0] = v_point
        sampled_points = [(x, int(last_p[x])) for x in range(0, width, 71) if x < width]

        for i, point in enumerate(sampled_points):
            line_points[l, i+1] = point

        output_dir = os.path.join('data', 'masks', 'cs2')
        os.makedirs(output_dir, exist_ok=True)
        m_filepath = os.path.join(output_dir, f'mask{l}.png')
        cv2.imwrite(m_filepath, m * 255)  # Save the mask, scaling to 0-255 for visibility
        print(f"Mask {l} saved to: {m_filepath}")

    else:
        print(f"Mask {l}: No non-zero pixels found")
        
        
# Create a DataFrame from the list of points
columns = ['v_point'] + [f'point{i}' for i in range(1, 11)]
df = pd.DataFrame(line_points, columns=columns)

csv_dir = os.path.join("data", "csv")
os.makedirs(csv_dir, exist_ok=True)
csv_filepath = os.path.join(csv_dir, 'line_points_cs2.csv')
df.to_csv(csv_filepath, index=False)