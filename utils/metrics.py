import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_metrics(train_losses, val_losses, train_accs, val_accs, epochs):
    
    # Plot training and validation metrics over epochs
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
     
    
# plot the image and the prediction
def plot_prediction(image, actual_mask, predicted_mask):
    
    # Plots the input image, the ground truth mask, and an overlay of ground truth and predicted masks.
    # Ground truth mask is shown in green, predicted mask in red, and overlap in yellow.
    
    # Prepare image
    img = image.cpu().squeeze().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)

    # Prepare masks
    gt_mask = actual_mask.cpu().squeeze().numpy()
    pred_mask = predicted_mask.cpu().squeeze().numpy()

    # Ensure masks are binary
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Create a white background and set yellow where mask==1
    gt_mask_rgb = np.ones((*gt_mask.shape, 3), dtype=np.float32)  # white background
    gt_mask_rgb[gt_mask == 1] = [1, 1, 0]  # yellow for foreground

    # Overlay: yellow for GT only, red for wrong prediction only, green for overlap
    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    overlay[(gt_mask == 1) & (pred_mask == 0)] = [1, 1, 0]
    overlay[(gt_mask == 0) & (pred_mask == 1)] = [1, 0, 0]
    overlay[(gt_mask == 1) & (pred_mask == 1)] = [0, 1, 0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_mask_rgb)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(img, alpha=0.7)
    axs[2].imshow(overlay, alpha=0.5)
    axs[2].set_title("Overlay: GT (Yellow), Wrong pred (Red), Overlap (Green)")
    axs[2].axis('off')

    # Legend
    yellow_patch = mpatches.Patch(color='yellow', label='Correct Label')
    red_patch = mpatches.Patch(color='red', label='Wrong Pixel')
    green_patch = mpatches.Patch(color='green', label='Correct Pixel')
    axs[2].legend(handles=[yellow_patch, red_patch, green_patch], loc='lower right')

    plt.tight_layout()
    plt.show()
