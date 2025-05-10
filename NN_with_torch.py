# Dependencies

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import tqdm
import matplotlib.pyplot as plt


# Setting `torch.manual_seed(42)` ensures that the random number generation in PyTorch is deterministic and reproducible. 
torch.cuda.empty_cache()
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# make sure to set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# define the the subclass of the torch.utils.data.Dataset class to use the dataset in the dataloader
class CustomDataset(Dataset):
    
    # When initialized, at least x should be passed to the dataset class
    # y may not be passed if the dataset is not used for supervised learning
    # transform is used to apply transformations to the data like normalization, augmentation, etc.
    
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    # __len__ method returns the number of samples in the dataset
    def __len__(self):
        return len(self.X)
    
    # __getitem__ method retrieves a sample from the dataset at the given index
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # with the transform applied if provided.
        if self.transform is not None:
            x = self.transform(x)
            
        # If y is None, it returns only x
        if self.y is not None:
            y = self.y[idx]
            return x, y
        return x

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, channels, strides):
        
        # Call the parent class constructor
        super(NeuralNetwork, self).__init__()
        
        # ReLU activation function
        self.relu = nn.ReLU()
  
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        
        # intial convolutional layer, followed by batch normalization
        # the input is a 3-channel image (RGB), and the output is 32 channels (filters)
        self.conv_init = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn_init = nn.BatchNorm2d(32)
        
        # Use ModuleList to store multiple layers of the same type in a list
        
        # Define depthwise (dw) separable convolution blocks
        self.dw_conv_layers = nn.ModuleList()
        self.dw_bn_layers = nn.ModuleList()
        # Define pointwise (pw) convolution blocks
        self.pw_conv_layers = nn.ModuleList()
        self.pw_bn_layers = nn.ModuleList()
        
        # Note that a depthwise separable convolution consists of a depthwise convolution followed by a pointwise convolution
        # The depthwise convolution applies a single filter to each input channel, while the pointwise convolution combines the outputs of the depthwise convolution.
        
        # network architecture after initial convolution and final fully connected layer
        in_channels = 32
        for i, (out_channels, stride) in enumerate(zip(channels, strides)):
            ##################################################################################################################################################
            # Depthwise convolution (separable by channel), followed by batch normalization
            self.dw_conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels))
            self.dw_bn_layers.append(nn.BatchNorm2d(in_channels))
            
            # Pointwise convolution (1x1 conv to change channels), followed by batch normalization
            self.pw_conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.pw_bn_layers.append(nn.BatchNorm2d(out_channels))
            
            in_channels = out_channels
        
        # Global average pooling --> replace with max pooling for lane detection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final fully connected layer to map to the number of classes
        self.final_fc = nn.Linear(channels[-1], num_classes)  # input is the last channel size
    
    
    # forward method defines the flow of data through the network
    # It takes the input x and passes it through the layers defined in __init__
    def forward(self, x):    

        # Reshape the input to (batch_size, 1, 28, 28)####################################################################################################
        x = x.view(x.size(0), 3, 32, 32)
        
        # Initial convolution (layer 1)
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu(x)
    
        # Iterate through the depthwise and pointwise convolution layers
        for i in range(len(self.dw_conv_layers)):
            
            # Depthwise convolution
            x = self.dw_conv_layers[i](x)
            x = self.dw_bn_layers[i](x)
            x = self.relu(x)
            
            # Pointwise convolution
            x = self.pw_conv_layers[i](x)
            x = self.pw_bn_layers[i](x)
            x = self.relu(x)
            
            # Apply dropout for regularization
            if i % 4 == 0:
                x = self.dropout(x)
        
        # Global average pooling and reshape to [batch_size, num_classes]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Map to the correct number of output classes
        x = self.final_fc(x)
        
        return x


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    
    # Saves the best validation value, when updated the model's state is saved
    best_val_acc = 0.0
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training phase
    for epoch in range(num_epochs):
        
        # set the model to training mode
        # eg activate dropout layers, batch normalization, etc. (disabled in eval mode)
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Wrap the train_loader with tqdm for progress tracking
        train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for inputs, labels in train_loop:
            
            # Move data to the device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients because gradients accumulate by default
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss using the criterion (loss function)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            
            # Compute the gradients of the loss with the respect to the model parameters,
            loss.backward()
            
            # update the model parameters using the optimizer
            optimizer.step()
            
            
            # Statistics
            
            # accumulate the loss for the current batch
            running_loss += loss.item() * inputs.size(0)
            
            # get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1) # _ is the max value, predicted is the index of the max value
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()       
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total
            
            # Update tqdm description with current metrics
            train_loop.set_description(f"Epoch {epoch+1}/{num_epochs} (Train) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        
        # Validation phase
        
        # set the model to evaluation mode
        model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # disable the gradient computation temporarily to save memory and computations (not needed for validation)
        with torch.no_grad():
            
            # Wrap the val_loader with tqdm for progress tracking
            val_loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)", leave=False)
            
            for inputs, labels in val_loop:
                
                # Move data to the device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass and compute the loss (no backward pass needed)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Compute the statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            # Update tqdm description with current metrics
            val_loop.set_description(f"Epoch {epoch+1}/{num_epochs} (Validation) Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Store metrics for plotting
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, 'f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc            
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, num_epochs)
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model


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
def plot_prediction(image, actual_label, predicted_label, class_names=None):
    
    # Convert image tensor to numpy and denormalize
    image = image.cpu().squeeze().permute(1, 2, 0).numpy()
    
    # Denormalize from [-1, 1] to [0, 1]
    image = (image * 0.5 + 0.5).clip(0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    
    # Get class names if available
    actual_class = f"Class {actual_label}" if class_names is None else class_names[actual_label]
    predicted_class = f"Class {predicted_label}" if class_names is None else class_names[predicted_label]
    
    title = f"Actual: {actual_class}\nPredicted: {predicted_class}"
    plt.title(title)
    plt.axis('off')
    plt.show()


# Main execution
def main():
    
    # Hyperparameters
    num_classes = 10
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 50
    
    ################################################################################################
    
    # Channel configuration for each block
    # Encoder path (increasing channels, decreasing spatial dimensions)
    encoder_channels = [32, 64, 128, 256]  
    encoder_strides = [2, 2, 2, 2]         

    # Bottleneck
    bottleneck_channels = [512]
    bottleneck_strides = [1]
       
    # Decoder path (decreasing channels)
    decoder_channels = [256, 128, 64, 32] # Gradually decrease channels
    decoder_strides = [2, 2, 2, 2]        # Upsample spatial dimensions by a factor of 2 at each step
     
    ################################################################################################# 
        
    # Combine encoder-bottleneck-decoder
    channels = encoder_channels + bottleneck_channels + decoder_channels
    strides = encoder_strides + bottleneck_strides + decoder_strides
    
    # Data loading and preprocessing, normalization reduces the com
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
    ])
    
    # Ensure the test value is not used in training
    test_sample = val_dataset[0]            # Reserve the first sample from the validation dataset
    val_dataset.data = val_dataset.data[1:] 
    val_dataset.targets = val_dataset.targets[1:]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize the model
    model = NeuralNetwork(num_classes, channels, strides).to(device)
    
    # Print model size information
    total_params = sum(p.numel() for p in model.parameters()) # Total number of parameters in the model
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Example of inference
    model.eval()
    with torch.no_grad():
        # Use the reserved test samples for inference
        example_input, actual_label = test_sample
        example_input = example_input.unsqueeze(0).to(device)  # Add batch dimension
        print(f"Actual label: {actual_label}")
        output = model(example_input)
        _, predicted = torch.max(output, 1)
        print(f"Predicted class: {predicted.item()}")
            
            # Plot the image and prediction
        try:
            plot_prediction(example_input[0], actual_label, predicted.item())
        except Exception as e:
            print(f"Could not plot image: {e}")


if __name__ == "__main__":
    main()