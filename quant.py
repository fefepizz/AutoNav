"""
Simple Model Compression Template for AutoNav Project

This module provides simple utilities for compressing pre-trained PyTorch models
to reduce memory usage and improve inference speed using FP16 and INT8 quantization.

Supports both half-precision (FP16) and INT8 quantization for model compression.
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import copy
import time
import numpy as np
from typing import Dict
import sys
import os
import os
import time
import numpy as np

# Import your models
from models.segNet import segNet
from utils.LoadDataset import LoadDataset


def compress_model_fp16(model: nn.Module, save_path: str = None):
    """
    Compress a pre-trained model using FP16 (half precision).
    
    Args:
        model: Pre-trained PyTorch model
        save_path: Optional path to save compressed model
    
    Returns:
        FP16 compressed model
    """
    print("Compressing model to FP16 (half precision)...")
    
    # Create a copy of the model
    compressed_model = copy.deepcopy(model)
    compressed_model.eval()
    
    # Convert to half precision (FP16) for smaller size and faster inference
    compressed_model = compressed_model.half()
    
    # Save compressed model if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(compressed_model.state_dict(), save_path)
        print(f"FP16 compressed model saved to: {save_path}")
    
    print("Model compression completed!")
    return compressed_model


def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  model_name: str = "Model"):
    """
    Simple evaluation of model performance.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        model_name: Name for logging
    """
    # Check model type
    is_fp16 = next(model.parameters()).dtype == torch.float16
    is_quantized = any('quantized' in str(type(module)).lower() for module in model.modules())
    
    if is_quantized:
        device = 'cpu'
        print(f"Note: INT8 quantized model will run on CPU only")
    elif is_fp16:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Note: FP16 model will run on {device.upper()}")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    num_batches = 0
    inference_times = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Evaluating {model_name} on {device.upper()}...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Convert data to match model precision
            if is_fp16:
                data = data.half()
                target = target.half()
            
            # Measure inference time
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_inference_time = np.mean(inference_times)
    
    print(f"{model_name} Results:")
    print(f"  Loss: {avg_loss:.6f}")
    print(f"  Avg Inference Time: {avg_inference_time*1000:.2f}ms")
    
    return avg_loss, avg_inference_time


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model size in various metrics.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict with size metrics including parameters and file size
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get parameter dtype
    param_dtype = next(model.parameters()).dtype
    
    # Check if model is quantized
    is_quantized = any('quantized' in str(type(module)).lower() for module in model.modules())
    
    # Calculate memory size based on precision
    if is_quantized:
        # INT8 quantized: 1 byte per parameter
        memory_bytes = total_params * 1
        precision = "INT8"
    elif param_dtype == torch.float16:
        # FP16: 2 bytes per parameter
        memory_bytes = total_params * 2
        precision = "FP16"
    elif param_dtype == torch.float32:
        # FP32: 4 bytes per parameter
        memory_bytes = total_params * 4
        precision = "FP32"
    else:
        # Default to FP32 calculation
        memory_bytes = total_params * 4
        precision = str(param_dtype)
    
    # Add buffer sizes
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_memory = memory_bytes + buffer_size
    
    # Convert to KB and MB
    memory_kb = total_memory / 1024
    memory_mb = memory_kb / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'precision': precision,
        'memory_bytes': total_memory,
        'memory_kb': memory_kb,
        'memory_mb': memory_mb,
        'is_quantized': is_quantized
    }


def compare_models(original_model: nn.Module, 
                  compressed_model: nn.Module, 
                  test_loader: DataLoader):
    """
    Compare original and compressed models with detailed metrics.
    """
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    # Evaluate both models
    orig_loss, orig_time = evaluate_model(original_model, test_loader, "Original")
    print("-" * 50)
    comp_loss, comp_time = evaluate_model(compressed_model, test_loader, "Compressed")
    
    # Model sizes with detailed metrics
    orig_size = get_model_size(original_model)
    comp_size = get_model_size(compressed_model)
    
    print("-" * 50)
    print("DETAILED SIZE METRICS:")
    print(f"Original Model ({orig_size['precision']}):")
    print(f"  Parameters: {orig_size['total_params']:,}")
    print(f"  Memory: {orig_size['memory_kb']:.1f} KB ({orig_size['memory_mb']:.1f} MB)")
    
    print(f"Compressed Model ({comp_size['precision']}):")
    print(f"  Parameters: {comp_size['total_params']:,}")
    print(f"  Memory: {comp_size['memory_kb']:.1f} KB ({comp_size['memory_mb']:.1f} MB)")
    print(f"  Quantized: {comp_size['is_quantized']}")
    
    # Compression ratios
    size_reduction = orig_size['memory_mb'] / comp_size['memory_mb']
    kb_saved = orig_size['memory_kb'] - comp_size['memory_kb']
    
    print("-" * 50)
    print("COMPRESSION SUMMARY:")
    print(f"Size Reduction: {size_reduction:.1f}x smaller")
    print(f"Memory Saved: {kb_saved:.1f} KB ({kb_saved/1024:.1f} MB)")
    print(f"Speed - Original: {orig_time*1000:.1f}ms, Compressed: {comp_time*1000:.1f}ms")
    
    if orig_time > 0 and comp_time > 0:
        speedup = orig_time / comp_time
        print(f"Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
    
    # Accuracy comparison
    accuracy_diff = abs(orig_loss - comp_loss)
    print(f"Accuracy Loss Difference: {accuracy_diff:.6f}")
    
    return {
        'original_size': orig_size,
        'compressed_size': comp_size,
        'compression_ratio': size_reduction,
        'kb_saved': kb_saved
    }


def save_compressed_model(model: nn.Module, filepath: str):
    """Save compressed model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Compressed model saved to: {filepath}")


def load_compressed_model(model_class, model_args: dict, filepath: str):
    """
    Load a compressed FP16 model.
    
    Args:
        model_class: Model class (segNet)
        model_args: Model initialization arguments  
        filepath: Path to saved compressed model
    """
    # Create model
    model = model_class(**model_args)
    model.eval()
    
    # Load weights and convert to FP16
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model = model.half()
    model.eval()
    
    print(f"Compressed model loaded from: {filepath}")
    return model


def prepare_test_data():
    """
    Prepare test data loaders for evaluation.
    Uses frame2 and frame3 for testing (same as validation data in train.py).
    """
    print("Preparing test data loaders...")
    
    # Simple transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Same data directories as train.py
    img_dir = "processed_data/img"
    mask_dir = "processed_data/mask"
    
    # Get all files
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

    # Test subset logic (same as validation in train.py)
    test_img_files = [f for f in img_files if os.path.basename(f).startswith(("frame2", "frame3"))]
    test_mask_files = [f for f in mask_files if os.path.basename(f).startswith(("frame2", "frame3"))]

    # Verify file matching
    assert len(test_img_files) == len(test_mask_files), "Mismatch between test images and masks"

    # Create dataset
    test_dataset = LoadDataset(test_img_files, test_mask_files, transform=transform)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print(f"Test data: {len(test_dataset)} samples")
    
    return test_loader


def compress_model_complete(model: nn.Module, 
                           model_name: str,
                           save_path: str = None):
    """
    Complete workflow to compress any pre-trained model.
    
    Args:
        model: Pre-trained PyTorch model to compress
        model_name: Name of the model for logging
        save_path: Optional path to save compressed model
    """
    print(f"\n{'='*50}")
    print(f"COMPRESSING {model_name.upper()}")
    print(f"{'='*50}")
    
    # Prepare test data
    test_loader = prepare_test_data()
    
    # Compress the model
    compressed_model = compress_model_fp16(model, save_path)
    
    # Compare models
    print(f"\nComparing {model_name} models...")
    compare_models(model, compressed_model, test_loader)
    
    return compressed_model


def quantize_model_int8(model: nn.Module, calibration_loader: DataLoader, num_calibration_batches: int = 50):
    """
    Quantize a pre-trained model using INT8 quantization with proper error handling.
    
    Args:
        model: Pre-trained PyTorch model
        calibration_loader: DataLoader with sample data for calibration
        num_calibration_batches: Number of batches to use for calibration
        
    Returns:
        INT8 quantized model or None if quantization fails
    """
    print("Starting INT8 quantization...")
    
    try:
        # Set quantization backend
        torch.backends.quantized.engine = 'fbgemm'
        
        # Prepare model for quantization
        model_fp32 = copy.deepcopy(model)
        model_fp32.eval()
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model_fp32, inplace=False)
        
        # Calibration phase - run model on sample data
        print("Calibrating INT8 model...")
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                try:
                    _ = model_prepared(data)
                except Exception as e:
                    print(f"Calibration failed at batch {i}: {e}")
                    return None
                
                if i % 10 == 0:
                    print(f"Calibration batch {i+1}/{min(num_calibration_batches, len(calibration_loader))}")
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        # Test the quantized model with a small sample
        print("Testing INT8 quantized model...")
        test_input = next(iter(calibration_loader))[0][:1]
        try:
            with torch.no_grad():
                _ = quantized_model(test_input)
            print("INT8 quantization test passed!")
        except Exception as test_error:
            print(f"ERROR: INT8 quantized model failed inference test!")
            print(f"Reason: {str(test_error)[:200]}...")
            return None
        
        print("INT8 quantization completed successfully!")
        return quantized_model
        
    except Exception as e:
        print(f"ERROR: INT8 quantization failed during preparation!")
        print(f"Reason: {str(e)[:200]}...")
        return None


def prepare_calibration_data():
    """
    Prepare calibration data loaders for INT8 quantization.
    Uses frame1 and frame4 for calibration (same as training data).
    """
    print("Preparing calibration data loaders...")
    
    # Simple transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Same data directories as train.py
    img_dir = "processed_data/img"
    mask_dir = "processed_data/mask"
    
    # Get all files
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

    # Calibration subset logic (same as training in train.py)
    calibration_img_files = [f for f in img_files if os.path.basename(f).startswith(("frame1", "frame4"))]
    calibration_mask_files = [f for f in mask_files if os.path.basename(f).startswith(("frame1", "frame4"))]

    # Verify file matching
    assert len(calibration_img_files) == len(calibration_mask_files), "Mismatch between calibration images and masks"

    # Create dataset
    calibration_dataset = LoadDataset(calibration_img_files, calibration_mask_files, transform=transform)

    # Create data loader
    calibration_loader = DataLoader(calibration_dataset, batch_size=4, shuffle=False)
    
    print(f"Calibration data: {len(calibration_dataset)} samples")
    
    return calibration_loader


if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install PyTorch: pip install torch torchvision")
        exit(1)
    
    print("Simple Model Compression for AutoNav")
    print("Compressing pre-trained models to reduce size and improve speed")
    print("Uses FP16 (half precision) for reliable model compression")
    print("-" * 60)
    
    # ===== COMPRESS SEGNET =====
    print("\n" + "="*60)
    print("COMPRESSING SEGNET MODEL")
    print("="*60)
    
    # Load pre-trained segNet
    segnet_model = segNet(n_channels=3)
    segnet_model.load_state_dict(torch.load('models/segNet_distilled.pth', map_location='cpu'))
    segnet_model.eval()
    print("Loaded pre-trained segNet model")
    
    # Compress segNet using FP16
    compressed_segnet = compress_model_complete(
        model=segnet_model,
        model_name="segNet",
        save_path='models/segNet_fp16.pth'
    )
    
    print(f"\n✅ COMPRESSION SUCCESSFUL for segNet")
    print("Compressed model saved successfully!")
    
    # ===== USAGE INSTRUCTIONS =====
    print("\nTO USE THIS COMPRESSION SCRIPT:")
    print("1. Ensure your pre-trained segNet model is in the models/ directory:")
    print("   - models/segNet_distilled.pth")
    print("2. Run the script to compress the segNet model")
    print("3. Compressed model will be saved as 'models/segNet_fp16.pth'")
    print("\nLOADING COMPRESSED MODEL:")
    print("model = segNet(n_channels=3)")
    print("model.load_state_dict(torch.load('models/segNet_fp16.pth'))")
    print("model = model.half()  # Convert to FP16")
    print("model.eval()")
    print("\nBENEFITS:")
    print("- ~50% smaller model size")
    print("- Potentially faster inference on compatible hardware")
    print("- Minimal accuracy loss")
