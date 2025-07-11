import cv2
import torch
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, img_files, mask_files, transform=None):
        def get_frame_number(filename):
            # Extract the frame number from filename like 'd5_s1_frame123.png' or 'mask123.png'
            import re
            if 'frame' in filename:
                match = re.search(r'frame(\d+)', filename)
            else:
                match = re.search(r'mask(\d+)', filename)
            return int(match.group(1)) if match else 0
            
        # Sort both lists numerically by frame/mask number
        self.img_files = sorted(img_files, key=get_frame_number)
        self.mask_files = sorted(mask_files, key=get_frame_number)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.img_files[idx])[:, :, ::-1]  # Convert BGR to RGB
            mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        except Exception as e:
            raise RuntimeError(f"Error loading image or mask at index {idx}: {e}")
            
        img = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
              
        return img, mask