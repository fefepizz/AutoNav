import torch
from utils.LoadDataset import LoadDataset
from utils.metrics import plot_prediction
from torch.utils.data import DataLoader


def evaluate(model, val_img_files, val_mask_files, batch_size, transform, device):
    
    
    
    val_dataset = LoadDataset(val_img_files, val_mask_files, transform=transform)    
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model.eval()
    with torch.no_grad():
        example_input, actual_label = next(iter(val_loader))
        example_input = example_input[0].unsqueeze(0).to(device)
        actual_label = actual_label[0].to(device)
        output = model(example_input)
        output = torch.sigmoid(output)
        predicted = (output > 0.5).float()

        try:
            plot_prediction(example_input[0], actual_label, predicted[0])
        except Exception as e:
            print(f"Could not plot image: {e}")