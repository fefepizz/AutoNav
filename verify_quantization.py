import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from models.MU_Net import MU_Net
from utils.LoadDataset import LoadDataset

def load_quantized_model(quantized_path, model):
    """
    Load and dequantize a quantized model following the instructions:
    1. Load the quantized state_dict
    2. Dequantize weights using stored quantization parameters
    3. Load into model with model.load_state_dict()
    """
    print(f"Loading quantized model from {quantized_path}")
    
    # 1. Load the quantized state_dict
    quantized_data = torch.load(quantized_path)
    quantization_params = quantized_data.pop('_quantization_params')
    
    # 2. Dequantize weights using stored quantization parameters
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
            # Keep non-quantized parameters as-is (bias terms, BatchNorm parameters)
            dequantized_state_dict[name] = param
    
    # 3. Load into model with model.load_state_dict()
    model.load_state_dict(dequantized_state_dict)
    return model

def calculate_metrics(outputs, labels):
    """Calculate IoU and pixel accuracy"""
    preds = (torch.sigmoid(outputs) > 0.5)
    labels_bin = (labels > 0.5)
    
    # IoU calculation
    intersection = (preds & labels_bin).float().sum()
    union = (preds | labels_bin).float().sum()
    iou = intersection / (union + 1e-8)
    
    # Pixel accuracy
    pixel_acc = (preds == labels_bin).float().mean()
    
    return iou.item(), pixel_acc.item()

def plot_prediction_with_metrics(image, actual_mask, original_pred, quantized_pred, 
                                original_metrics, quantized_metrics, efficiency_metrics):
    """
    Plot prediction comparison with metrics overlay
    """
    # Prepare image
    img = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)

    # Prepare masks
    gt_mask = actual_mask.cpu().squeeze().numpy()
    orig_mask = original_pred.cpu().squeeze().numpy()
    quant_mask = quantized_pred.cpu().squeeze().numpy()

    # Ensure masks are binary
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    orig_mask = (orig_mask > 0.5).astype(np.uint8)
    quant_mask = (quant_mask > 0.5).astype(np.uint8)

    # Create overlays
    def create_overlay(gt, pred):
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        overlay[(gt == 1) & (pred == 0)] = [1, 1, 0]  # GT only: yellow
        overlay[(gt == 0) & (pred == 1)] = [1, 0, 0]  # Prediction only: red
        overlay[(gt == 1) & (pred == 1)] = [0, 1, 0]  # Overlap: green
        return overlay

    orig_overlay = create_overlay(gt_mask, orig_mask)
    quant_overlay = create_overlay(gt_mask, quant_mask)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

    # Top row: Input image and ground truth
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title("Input Image", fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    gt_mask_rgb = np.ones((*gt_mask.shape, 3), dtype=np.float32)
    gt_mask_rgb[gt_mask == 1] = [1, 1, 0]  # yellow for foreground
    ax2.imshow(gt_mask_rgb)
    ax2.set_title("Ground Truth Mask", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Second row: Predictions with overlays
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(img, alpha=0.7)
    ax3.imshow(orig_overlay, alpha=0.6)
    ax3.set_title("Original MU_Net Prediction", fontsize=14, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(img, alpha=0.7)
    ax4.imshow(quant_overlay, alpha=0.6)
    ax4.set_title("Quantized MU_Net Prediction", fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Legend
    yellow_patch = mpatches.Patch(color='yellow', label='GT Only')
    red_patch = mpatches.Patch(color='red', label='Prediction Only')
    green_patch = mpatches.Patch(color='green', label='Overlap')
    ax4.legend(handles=[yellow_patch, red_patch, green_patch], loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # Metrics text box (spans right side)
    ax_metrics = fig.add_subplot(gs[:, 2:])
    ax_metrics.axis('off')
    
    # Create metrics text
    metrics_text = f"""
MODEL COMPARISON METRICS

PERFORMANCE METRICS:
┌─────────────────────────────────────────┐
│ Metric          │ Original │ Quantized │
├─────────────────────────────────────────┤
│ IoU             │ {original_metrics['iou']:.4f}   │ {quantized_metrics['iou']:.4f}    │
│ Pixel Accuracy  │ {original_metrics['acc']:.4f}   │ {quantized_metrics['acc']:.4f}    │
│ Loss            │ {original_metrics['loss']:.4f}   │ {quantized_metrics['loss']:.4f}    │
│ Inference (ms)  │ {original_metrics['time']*1000:.1f}     │ {quantized_metrics['time']*1000:.1f}      │
└─────────────────────────────────────────┘

EFFICIENCY GAINS:
┌─────────────────────────────────────────┐
│ Model Size Reduction: {efficiency_metrics['size_reduction']:.1f}%             │
│ Performance Loss (IoU): {efficiency_metrics['performance_loss_iou']:.2f}%       │
│ Performance Loss (Acc): {efficiency_metrics['performance_loss_acc']:.2f}%       │
│ Speed Change: {efficiency_metrics['speed_change']:+.1f}%                  │
└─────────────────────────────────────────┘

QUANTIZATION ASSESSMENT:
{efficiency_metrics['assessment']}

SUMMARY:
• Original Model: {efficiency_metrics['original_size']:.1f} MB
• Quantized Model: {efficiency_metrics['quantized_size']:.1f} MB
• Compression Ratio: {efficiency_metrics['original_size']/efficiency_metrics['quantized_size']:.1f}:1

PIXEL ANALYSIS:
• True Positives: {np.sum((gt_mask == 1) & (quant_mask == 1)):,} pixels
• False Positives: {np.sum((gt_mask == 0) & (quant_mask == 1)):,} pixels  
• False Negatives: {np.sum((gt_mask == 1) & (quant_mask == 0)):,} pixels
• True Negatives: {np.sum((gt_mask == 0) & (quant_mask == 0)):,} pixels
"""
    
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle("MU_Net Quantization Verification Results", fontsize=16, fontweight='bold', y=0.95)
    
    return fig

def evaluate_model(model, eval_loader, criterion, device, model_name):
    """Evaluate a model and return metrics"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_time = 0.0
    count = 0
    
    print(f"Evaluating {model_name}...")
    
    with torch.no_grad():
        eval_loop = tqdm.tqdm(eval_loader, desc=f"Evaluating {model_name}", leave=True)
        for inputs, labels in eval_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            # Calculate metrics
            iou, pixel_acc = calculate_metrics(outputs, labels)
            total_iou += iou
            total_acc += pixel_acc
            total_time += inference_time
            count += 1
            
            eval_loop.set_description(f"Evaluating {model_name} - Loss: {loss.item():.4f}, IoU: {iou:.4f}, Acc: {pixel_acc:.4f}")
    
    avg_loss = total_loss / len(eval_loader.dataset)
    avg_iou = total_iou / count
    avg_acc = total_acc / count
    avg_time = total_time / count
    
    return avg_loss, avg_iou, avg_acc, avg_time

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading setup (similar to train.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_dir = "processed_data/img"
    mask_dir = "processed_data/mask"
    
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
    
    # Use all available data for evaluation (similar to validation split in train.py)
    eval_img_files = [f for f in img_files if os.path.basename(f).startswith(("frame2", "frame3", "frame5", "frame6"))]
    eval_mask_files = [f for f in mask_files if os.path.basename(f).startswith(("frame2", "frame3", "frame5", "frame6"))]
    
    print(f"Found {len(eval_img_files)} evaluation images")
    print(f"Found {len(eval_mask_files)} evaluation masks")
    
    assert len(eval_img_files) == len(eval_mask_files), "Mismatch between evaluation images and masks"
    
    eval_dataset = LoadDataset(eval_img_files, eval_mask_files, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Load original MU_Net model
    print("\n" + "="*60)
    print("Loading original MU_Net model...")
    original_model = MU_Net(n_channels=3).to(device)
    original_model_path = "models/MU_Net_distilled.pth"
    
    if not os.path.exists(original_model_path):
        print(f"Error: Original model not found at {original_model_path}")
        return
    
    original_state_dict = torch.load(original_model_path, map_location=device)
    original_model.load_state_dict(original_state_dict)
    
    # Print model summary
    total_params = sum(p.numel() for p in original_model.parameters())
    total_trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024 * 1024)
    print(f"Original Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {total_trainable_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # Load quantized MU_Net model
    print("\nLoading quantized MU_Net model...")
    quantized_model = MU_Net(n_channels=3).to(device)
    quantized_model_path = "quantized/MU_Net_distilled_quantized.pth"
    
    if not os.path.exists(quantized_model_path):
        print(f"Error: Quantized model not found at {quantized_model_path}")
        return
    
    quantized_model = load_quantized_model(quantized_model_path, quantized_model)
    
    # Evaluate both models
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Evaluate original model
    original_loss, original_iou, original_acc, original_time = evaluate_model(
        original_model, eval_loader, criterion, device, "Original MU_Net"
    )
    
    # Evaluate quantized model
    quantized_loss, quantized_iou, quantized_acc, quantized_time = evaluate_model(
        quantized_model, eval_loader, criterion, device, "Quantized MU_Net"
    )
    
    # Display results comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Original':<12} {'Quantized':<12} {'Difference':<15}")
    print("-" * 65)
    print(f"{'Loss':<20} {original_loss:<12.4f} {quantized_loss:<12.4f} {quantized_loss-original_loss:<+15.4f}")
    print(f"{'IoU':<20} {original_iou:<12.4f} {quantized_iou:<12.4f} {quantized_iou-original_iou:<+15.4f}")
    print(f"{'Pixel Accuracy':<20} {original_acc:<12.4f} {quantized_acc:<12.4f} {quantized_acc-original_acc:<+15.4f}")
    print(f"{'Inference Time (ms)':<20} {original_time*1000:<12.1f} {quantized_time*1000:<12.1f} {(quantized_time-original_time)*1000:<+15.1f}")
    print("-" * 65)
    
    # Model sizes
    original_size = os.path.getsize(original_model_path) / 1024 / 1024
    quantized_size = os.path.getsize(quantized_model_path) / 1024 / 1024
    size_reduction = (1 - quantized_size / original_size) * 100
    
    print(f"{'Model Size (MB)':<20} {original_size:<12.1f} {quantized_size:<12.1f} {-size_reduction:<+15.1f}%")
    print("="*60)
    
    # Summary
    performance_loss_iou = abs(quantized_iou - original_iou) / original_iou * 100
    performance_loss_acc = abs(quantized_acc - original_acc) / original_acc * 100
    speed_change = ((quantized_time/original_time - 1) * 100)
    
    print(f"\nSummary:")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Performance loss (IoU): {performance_loss_iou:.2f}%")
    print(f"  Performance loss (Accuracy): {performance_loss_acc:.2f}%")
    print(f"  Speed change: {speed_change:+.1f}%")
    
    if performance_loss_iou < 5.0 and size_reduction > 50.0:
        assessment = "✓ Quantization successful: Good compression with minimal performance loss"
        print(f"  {assessment}")
    elif performance_loss_iou < 10.0:
        assessment = "⚠ Quantization acceptable: Moderate performance loss"
        print(f"  {assessment}")
    else:
        assessment = "✗ Quantization may need tuning: Significant performance loss"
        print(f"  {assessment}")
    
    # Generate prediction plot with metrics
    print(f"\nGenerating prediction comparison plot...")
    
    # Get a sample for visualization
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        # Take the first sample from evaluation data
        sample_input, sample_label = next(iter(eval_loader))
        sample_input = sample_input.to(device)
        sample_label = sample_label.to(device)
        
        # Get predictions from both models
        original_output = original_model(sample_input)
        quantized_output = quantized_model(sample_input)
        
        # Convert to predictions
        original_pred = torch.sigmoid(original_output) > 0.5
        quantized_pred = torch.sigmoid(quantized_output) > 0.5
        
        # Calculate metrics for this specific sample
        sample_orig_iou, sample_orig_acc = calculate_metrics(original_output, sample_label)
        sample_quant_iou, sample_quant_acc = calculate_metrics(quantized_output, sample_label)
        sample_orig_loss = criterion(original_output, sample_label).item()
        sample_quant_loss = criterion(quantized_output, sample_label).item()
        
        # Prepare metrics dictionaries
        original_metrics = {
            'iou': sample_orig_iou,
            'acc': sample_orig_acc,
            'loss': sample_orig_loss,
            'time': original_time
        }
        
        quantized_metrics = {
            'iou': sample_quant_iou,
            'acc': sample_quant_acc,
            'loss': sample_quant_loss,
            'time': quantized_time
        }
        
        efficiency_metrics = {
            'size_reduction': size_reduction,
            'performance_loss_iou': performance_loss_iou,
            'performance_loss_acc': performance_loss_acc,
            'speed_change': speed_change,
            'assessment': assessment,
            'original_size': original_size,
            'quantized_size': quantized_size
        }
        
        # Create the plot
        fig = plot_prediction_with_metrics(
            sample_input[0], sample_label[0], 
            original_pred[0], quantized_pred[0],
            original_metrics, quantized_metrics, efficiency_metrics
        )
        
        # Save the plot
        output_filename = "MU_Net_quantization_verification.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Prediction comparison plot saved as: {output_filename}")

if __name__ == "__main__":
    main()
