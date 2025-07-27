import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from models.MU_Net import MU_Net
    
def quantize_model(model_path):
    
    os.makedirs("quantized", exist_ok=True)
    
    # Create model
    model = MU_Net(n_channels=3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    # Manual quantization since grouped convs aren't supported by dynamic quantization
    print("Applying manual quantization to model weights...")
    
    def quantize_weights_to_int8(model):
        """Manually quantize model weights to int8"""
        quantized_state_dict = {}
        quantization_params = {}
        conv_params = 0
        bn_params = 0
        
        for name, param in model.state_dict().items():
            if param.dtype == torch.float32 and len(param.shape) > 1:  # Weight tensors (Conv2d)
                # Calculate scale and zero point for quantization
                min_val = param.min().item()
                max_val = param.max().item()
                
                # Avoid division by zero
                if max_val == min_val:
                    scale = 1.0
                    zero_point = 0
                else:
                    scale = (max_val - min_val) / 255.0
                    zero_point = int(-min_val / scale)
                    zero_point = max(0, min(255, zero_point))
                
                # Quantize to int8 and store as int8
                quantized = torch.round((param - min_val) / scale).clamp(0, 255)
                quantized_int8 = (quantized - 128).to(torch.int8)
                
                # Store quantized weights as int8
                quantized_state_dict[name] = quantized_int8
                
                # Store quantization parameters for reconstruction
                quantization_params[name] = {
                    'scale': scale,
                    'min_val': min_val,
                    'zero_point': zero_point
                }
                
                conv_params += param.numel()
                print(f"  Quantized {name}: {param.shape} ({param.numel()} params)")
            else:
                # Keep bias terms and BatchNorm parameters as float32
                quantized_state_dict[name] = param
                if 'batch_norm' in name or 'weight' in name or 'bias' in name:
                    bn_params += param.numel()
        
        print(f"\nParameter breakdown:")
        print(f"  Conv2d weights (quantized): {conv_params:,} parameters")
        print(f"  BatchNorm + bias (float32): {bn_params:,} parameters")
        print(f"  Quantization impact: {conv_params / (conv_params + bn_params) * 100:.1f}% of total params")
        
        # Save quantization parameters separately
        quantized_state_dict['_quantization_params'] = quantization_params
        return quantized_state_dict
    
    # Apply manual quantization
    quantized_state_dict = quantize_weights_to_int8(model)
    print()
    
    # Save quantized model
    model_name = Path(model_path).stem
    output_path = os.path.join("quantized", f"{model_name}_quantized.pth")
    torch.save(quantized_state_dict, output_path)
    
    # Show results
    original_size = os.path.getsize(model_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Original: {original_size:.1f} MB")
    print(f"Quantized: {quantized_size:.1f} MB")
    print(f"Reduction: {reduction:.1f}%")
    print(f"Saved to: {output_path}")
    print(f"Note: Quantized weights are stored as int8. To load:")
    print(f"  1. Load the quantized state_dict")
    print(f"  2. Dequantize weights using stored quantization parameters")
    print(f"  3. Load into model with model.load_state_dict()")
    print()

if __name__ == "__main__":
    # Quantize segNet_distilled only
    model_path = "models/segNet_distilled.pth"
    
    if os.path.exists(model_path):
        print("PyTorch Model Quantization - segNet_distilled")
        quantize_model(model_path)
    else:
        print(f"Model not found: {model_path}")
        print("Make sure segNet_distilled.pth exists in the models/ directory")