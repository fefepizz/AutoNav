#!/usr/bin/env python3
"""Test script to compare original vs quantized model performance"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import time
import numpy as np
from tqdm import tqdm

# Add models to path
import sys
sys.path.append('models')

from models.MU_Net.MU_Net_model import segNet
from utils.LoadDataset import LoadDataset

def load_quantized_model(quantized_path, model):
    """Load and dequantize a quantized model"""
    print(f"Loading quantized model from {quantized_path}")
    
    # Load quantized state dict
    quantized_data = torch.load(quantized_path)
    quantization_params = quantized_data.pop('_quantization_params')
    
    # Dequantize weights
    dequantized_state_dict = {}
    for name, param in quantized_data.items():
        if name in quantization_params:
            # Dequantize int8 back to float32
            scale = quantization_params[name]['scale']
            min_val = quantization_params[name]['min_val']
            
            # Convert int8 back to original range
            dequantized = (param.float() + 128) * scale + min_val
            dequantized_state_dict[name] = dequantized
        else:
            # Keep non-quantized parameters as-is
            dequantized_state_dict[name] = param
    
    # Load into model
    model.load_state_dict(dequantized_state_dict)
    return model

def calculate_metrics(outputs, targets):
    """Calculate IoU and pixel accuracy"""
    outputs = torch.sigmoid(outputs) > 0.5
    targets = targets > 0.5
    
    intersection = (outputs & targets).float().sum()
    union = (outputs | targets).float().sum()
    
    iou = intersection / (union + 1e-8)
    pixel_acc = (outputs == targets).float().mean()
    
    return iou.item(), pixel_acc.item()

def test_model(model, test_loader, device, model_name):
    """Test a model and return metrics"""
    model.eval()
    total_iou = 0
    total_acc = 0
    total_time = 0
    count = 0
    
    print(f"Testing {model_name}...")
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"{model_name}"):
            images = images.to(device)
            masks = masks.to(device).float()
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            iou, pixel_acc = calculate_metrics(outputs, masks)
            
            total_iou += iou
            total_acc += pixel_acc
            total_time += inference_time
            count += 1
    
    avg_iou = total_iou / count
    avg_acc = total_acc / count
    avg_time = total_time / count
    
    return avg_iou, avg_acc, avg_time

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load test data (frame1 for testing)
    img_dir = "processed_data/img"
    mask_dir = "processed_data/mask"
    
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
    
    # Get frame numbers for matching
    def get_frame_number(filename):
        import re
        # For images: frame1_100.png -> 100
        # For masks: frame1_100_mask.png -> 100
        match = re.search(r'frame\d+_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    # Filter for frame1 images and find matching masks
    test_img_files = [f for f in img_files if os.path.basename(f).startswith("frame1")]
    test_mask_files = []
    
    for img_file in test_img_files:
        # Extract the frame number (e.g., frame1_100.png -> 100)
        frame_num = get_frame_number(os.path.basename(img_file))
        # Look for corresponding mask: frame1_100_mask.png
        matching_mask = f"processed_data/mask/frame1_{frame_num}_mask.png"
        if os.path.exists(matching_mask):
            test_mask_files.append(matching_mask)
        else:
            # Remove this image if no matching mask
            test_img_files.remove(img_file)
    
    print(f"Found {len(test_img_files)} frame1 test images with matching masks")
    
    # If no frame1, try frame2, frame5, frame6 for more test data
    if len(test_img_files) == 0:
        print("No frame1 test images found! Trying other frames...")
        for frame_prefix in ["frame2", "frame5", "frame6", "frame4"]:
            candidate_imgs = [f for f in img_files if os.path.basename(f).startswith(frame_prefix)]
            candidate_masks = []
            
            for img_file in candidate_imgs:
                frame_num = get_frame_number(os.path.basename(img_file))
                matching_mask = f"processed_data/mask/{frame_prefix}_{frame_num}_mask.png"
                if os.path.exists(matching_mask):
                    candidate_masks.append(matching_mask)
                else:
                    candidate_imgs.remove(img_file)
            
            if len(candidate_imgs) > 0:
                test_img_files.extend(candidate_imgs)
                test_mask_files.extend(candidate_masks)
                print(f"Added {len(candidate_imgs)} {frame_prefix} images")
        
        print(f"Total test images: {len(test_img_files)}")
    
    test_dataset = LoadDataset(test_img_files, test_mask_files, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load original model
    print("Loading original model...")
    original_model = segNet(n_channels=3).to(device)
    original_state_dict = torch.load("models/segNet_distilled.pth")
    original_model.load_state_dict(original_state_dict)
    
    # Load quantized model
    print("Loading quantized model...")
    quantized_model = segNet(n_channels=3).to(device)
    quantized_model = load_quantized_model("quantized/segNet_distilled_quantized.pth", quantized_model)
    
    # Test both models
    print("\n" + "="*50)
    original_iou, original_acc, original_time = test_model(original_model, test_loader, device, "Original")
    quantized_iou, quantized_acc, quantized_time = test_model(quantized_model, test_loader, device, "Quantized")
    
    # Results
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"{'Metric':<20} {'Original':<12} {'Quantized':<12} {'Difference':<12}")
    print("-" * 56)
    print(f"{'IoU':<20} {original_iou:<12.4f} {quantized_iou:<12.4f} {quantized_iou-original_iou:<+12.4f}")
    print(f"{'Pixel Accuracy':<20} {original_acc:<12.4f} {quantized_acc:<12.4f} {quantized_acc-original_acc:<+12.4f}")
    print(f"{'Inference Time (ms)':<20} {original_time*1000:<12.1f} {quantized_time*1000:<12.1f} {(quantized_time-original_time)*1000:<+12.1f}")
    print("-" * 56)
    
    # Model sizes
    original_size = os.path.getsize("models/segNet_distilled.pth") / 1024 / 1024
    quantized_size = os.path.getsize("quantized/segNet_distilled_quantized.pth") / 1024 / 1024
    size_reduction = (1 - quantized_size / original_size) * 100
    
    print(f"{'Model Size (MB)':<20} {original_size:<12.1f} {quantized_size:<12.1f} {-size_reduction:<+12.1f}%")
    print("="*50)
    
    # Summary
    performance_loss = abs(quantized_iou - original_iou) / original_iou * 100
    print(f"\nSummary:")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Performance loss: {performance_loss:.2f}% (IoU)")
    print(f"  Speed change: {((quantized_time/original_time - 1) * 100):+.1f}%")

if __name__ == "__main__":
    main()
