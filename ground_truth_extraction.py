import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
import os
import subprocess

print("CUDA is available:", torch.cuda.is_available())

def ensure_sam_model(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.isfile(model_path):
        print("SAM model not found, downloading...")
        model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        subprocess.run(["wget", "-O", model_path, model_url], check=True)
        print("Download complete.")

def load_images(images_path, start_idx=0, end_idx=None):
    def get_frame_number(filename):
        # Extract the frame number from filename like 'd5_s1_frame123.png'
        import re
        match = re.search(r'frame(\d+)', filename)
        return int(match.group(1)) if match else 0

    image_files = sorted(os.listdir(images_path), key=get_frame_number)
    if end_idx is None:
        end_idx = len(image_files)
    image_files = image_files[start_idx:end_idx]
    images = []
    for img_file in image_files:
        image_full_path = os.path.join(images_path, img_file)
        img = cv2.imread(image_full_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
    return images

def save_line_points_csv(line_points, csv_filepath):
    columns = ['v_point'] + [f'point{i}' for i in range(1, 11)]
    df = pd.DataFrame(line_points, columns=columns)
    csv_dir = os.path.dirname(csv_filepath)
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(csv_filepath, index=False)

def process_mask(mask, mask_index, output_dir):
    """
    Process a single mask to find line points and save the mask.
    
    Args:
        mask (np.array): Binary mask array
        mask_index (int): Index of the current mask (1-based)
        output_dir (str): Directory where to save the processed mask
    
    Returns:
        tuple: (v_point, sampled_points) or (None, None) if no valid points found
    """
    height, width = mask.shape
    last_points = np.zeros(width, dtype=int)
    
    # Find the highest point (lowest y value) for each x coordinate
    for x in range(width):
        for y in range(height - 1):
            if mask[y][x] == 1:
                last_points[x] = y
                break
    
    min_val = height
    for val in last_points:
        if 0 < val < min_val:
            min_val = val
            
    # Validate mask based on minimum value and mean
    if not (20 < min_val < height and np.mean(mask) > 0.25):  # Adjusted for thinner stems
        print(f"Mask {mask_index}: No valid points found")
        return None, None
        
    # Find the median x coordinate for points at minimum y value
    indices = [i for i, val in enumerate(last_points) if val == min_val]
    median_x = indices[len(indices) // 2]
    v_point = (int(median_x), int(min_val))
    print(f"Mask {mask_index}, v_point: {v_point}")
    
    # Sample points along the line
    SAMPLE_INTERVAL = 71  # Distance between sampled points
    sampled_points = [(x, int(last_points[x])) for x in range(0, width, SAMPLE_INTERVAL) if x < width]
    
    # Save the mask
    os.makedirs(output_dir, exist_ok=True)
    mask_filepath = os.path.join(output_dir, f'mask{mask_index}.png')
    cv2.imwrite(mask_filepath, mask * 255)
    print(f"Mask {mask_index} saved to: {mask_filepath}")
    
    return v_point, sampled_points

def main():
    """Main function to process images and generate masks."""
    # Load images
    img_dir = os.path.join("data", "TinyAgri", "Crops", "scene1") ########################################
    images = load_images(img_dir)

    # Initialize SAM model
    model_path = os.path.join("models/sam_model", "sam_vit_h_4b8939.pth")
    ensure_sam_model(model_path)
    
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)

    # Configure mask generator
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=4,
        min_mask_region_area=4000, 
    )

    # Generate masks for each image
    masks = []
    original_indices = []  # Track original image indices
    for img_idx, img in enumerate(images, start=1):  # Start indexing from 1
        anns = mask_gen.generate(img)
        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        if len(sorted_anns) == 0:  # Simple check to avoid IndexError
            print(f"No masks found for image {img_idx}, skipping...")
            continue
        ann = sorted_anns[0]
        masks.append(ann['segmentation'].astype(np.uint8))
        original_indices.append(img_idx)  # Keep track of original image index
        print(f"Done with mask: {img_idx}")

    # Process masks and collect line points
    line_points = np.zeros((len(masks), 11), dtype=object)
    output_dir = os.path.join('data', 'masks', 'Crops', 'scene1') ########################################
    
    for mask_idx, (mask, orig_idx) in enumerate(zip(masks, original_indices)):
        v_point, sampled_points = process_mask(mask, orig_idx, output_dir)  # Use original index for naming
        if v_point and sampled_points:
            line_points[mask_idx, 0] = v_point  # Use mask_idx for array indexing
            for point_idx, point in enumerate(sampled_points):
                line_points[mask_idx, point_idx + 1] = point

    # Save results
    csv_filepath = os.path.join("data", "csv", "line_points_cs1.csv") #############################################
    save_line_points_csv(line_points, csv_filepath)

if __name__ == "__main__":
    main()