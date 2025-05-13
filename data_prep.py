import os
import cv2
from glob import glob


# import all the images from the folders
f1 = os.path.join("TinyAgri/Tomatoes/", "scene1")
f2 = os.path.join("TinyAgri/Tomatoes/", "scene2")
f3 = os.path.join("TinyAgri/Crops/", "scene1")
f4 = os.path.join("TinyAgri/Crops/", "scene2")

# import all the masks from the folders
m1 = os.path.join("masks/", "ts1")
m2 = os.path.join("masks/", "ts2")
m3 = os.path.join("masks/", "cs1")
m4 = os.path.join("masks/", "cs2")

# Create output directory
output_dir = "processed_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process images and masks
def process_data(img_folder, mask_folder, folder_idx):
    
    # Get all image files
    img_files = sorted(glob(os.path.join(img_folder, "*.png")))
    print(f"Found {len(img_files)} images in {img_folder}")
    
    # Get all mask files (for debugging)
    mask_files = sorted(glob(os.path.join(mask_folder, "*.png")))
    print(f"Found {len(mask_files)} masks in {mask_folder}")
    
    pairs = []
    
    # if the image has a corresponding mask, then process them
    for img_path in img_files:
        
        # Get the image name
        img_name = os.path.basename(img_path) 
        
        try:
                        
            # This line extracts the frame number from the image filename
            # Split at 'frame' and take the second part 
            # then split at '.' to get the number (first part)
            frame_num = int(img_name.split('frame')[1].split('.')[0])
            
            # Construct the corresponding mask name
            mask_name = f"mask{frame_num}.png"
            mask_path = os.path.join(mask_folder, mask_name)
            
            # Check if the mask exists
            if os.path.exists(mask_path):
                
                # Read image and mask
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    
                    # reshape the image to 80x80
                    img_resized = cv2.resize(img, (80, 80))
                    
                    # reshape the mask to 80x80
                    mask_resized = cv2.resize(mask, (80, 80))
                    
                    # format the name as frame{i}_{j}.png
                    formatted_name = f"frame{folder_idx}_{frame_num}.png"
                    
                    # associate the image with the mask and formatted name
                    pairs.append((img_resized, mask_resized, formatted_name))
                    print(f"Processed image {frame_num}")
                    
                else:
                    print(f"something went wrong with {img_name}")
                
            else:
                print(f"Mask not found for image {frame_num}")
                
        except Exception as e:
            print(f"Could not process {img_name}: {e}")
    
    return pairs

# Process all folders
data = []
data.extend(process_data(f1, m1, 1))
data.extend(process_data(f2, m2, 2))
data.extend(process_data(f3, m3, 3))
data.extend(process_data(f4, m4, 4))


# put them in the same folder
for i, (img, mask, name) in enumerate(data):
    
    # name is formatted as frame{i}_{j}.png
    # remove the last 4 characters to get the base name (.png)
    base_name = os.path.join(output_dir, name[:-4])
    
    # Save the processed image
    img_path = os.path.join(output_dir, f"{base_name}.png")
    cv2.imwrite(img_path, img)
    
    # Save the processed mask
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask)


# Create img and mask subdirectories
img_dir = os.path.join(output_dir, "img")
mask_dir = os.path.join(output_dir, "mask")
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

for i, (img, mask, name) in enumerate(data):
    # Save the processed image
    img_path = os.path.join(output_dir, "img", name)  # Use name directly
    cv2.imwrite(img_path, img)
    
    # Save the processed mask
    mask_name = name.replace(".png", "_mask.png")  # Append "_mask" to the name
    mask_path = os.path.join(output_dir, "mask", mask_name)
    cv2.imwrite(mask_path, mask)
    
    print(f"Saving image to: {img_path}")
    print(f"Saving mask to: {mask_path}")


print(f"Processed {len(data)} image-mask pairs and saved to {output_dir}")