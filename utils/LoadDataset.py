import cv2
import torch
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    def __init__(self, img_files, mask_files, transform=None):
        self.img_files = img_files
        self.mask_files = mask_files
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