import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import copy
from tqdm import tqdm
from NN_with_torch import NeuralNetwork  # Import NeuralNetwork at the top level

"""
Simple Neural Architecture Search (NAS) Implementation

This script implements a basic neural architecture search that:
1. Generates random CNN architectures with varying depths, channels, and strides
2. Evaluates architectures using activation diversity as a proxy metric for performance
3. Tracks the best architectures found during the search process

Main components:
- ActivationHook: Captures layer activations during forward passes
- calculate_activation_score: Evaluates architecture quality using activation covariance
- generate_random_architecture: Creates random CNN architectures
- run_neural_architecture_search: Main search algorithm that samples and evaluates architectures
- main: Sets up data and executes the search
"""

class ActivationHook:
    """Hook to capture activations from network layers
    
    This class registers hooks to neural network modules and stores their 
    activation outputs during forward passes. It's used to analyze the 
    internal representations learned by the network.
    """
    
    def __init__(self):
        # Dictionary to store activations from different layers
        self.activations = {}
        # List to keep track of registered hooks for later removal
        self.hooks = []
    
    def register_hook(self, name, module):
        """Register a forward hook on the module
        
        Args:
            name (str): Identifier for the module
            module (nn.Module): Layer to hook into
        """
        hook = module.register_forward_hook(
            lambda module, input, output: self.activations.update({name: output.detach()})
        )
        self.hooks.append(hook)
    
    def clear(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

def calculate_activation_score(model, data_loader, num_batches=10):
    """
    Calculate a score based on the covariance of activations.
    A good architecture should have diverse activations across different layers.
    
    This function:
    1. Registers hooks to capture activations from Conv2d and Linear layers
    2. Processes a subset of data through the model
    3. Analyzes activation patterns using eigenvalue statistics
    4. Returns a score where higher values indicate better feature diversity
    
    Args:
        model: The neural network to evaluate
        data_loader: DataLoader providing input samples
        num_batches: Number of batches to process for evaluation
        
    Returns:
        float: Architecture quality score based on activation diversity
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Register hooks for all convolutional and linear layers
    activation_hook = ActivationHook()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            activation_hook.register_hook(name, module)
    
    # Forward pass through a subset of data
    batch_idx = 0
    all_covs = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            if batch_idx >= num_batches:
                break
                
            inputs = inputs.to(device)
            model(inputs)  # Forward pass
            
            # Process activations
            batch_covs = []
            for name, activation in activation_hook.activations.items():
                # Reshape activations and calculate covariance
                act = activation.reshape(activation.size(0), -1)  # Flatten all dimensions except batch
                
                # Skip if activations have too few features
                if act.size(1) <= 1:
                    continue
                    
                # Calculate covariance matrix
                act = act - act.mean(dim=0)
                cov = torch.mm(act.t(), act) / (act.size(0) - 1)
                
                # Use the normalized variance of eigenvalues as a diversity measure
                try:
                    eigenvalues = torch.linalg.eigvalsh(cov)
                    # Filter out negative eigenvalues (numerical errors)
                    eigenvalues = eigenvalues[eigenvalues > 1e-10]
                    if len(eigenvalues) > 0:
                        # Normalized variance of eigenvalues - higher is better for diverse representations
                        normalized_variance = eigenvalues.std() / (eigenvalues.mean() + 1e-8)
                        batch_covs.append(normalized_variance.item())
                except Exception as e:
                    # Skip layers where eigenvalue calculation fails
                    continue
            
            if batch_covs:
                all_covs.append(np.mean(batch_covs))
            batch_idx += 1
    
    # Clean up hooks
    activation_hook.clear()
    
    if not all_covs:
        return 0.0
    
    # Higher score means better architecture (more diverse activations)
    return np.mean(all_covs)

def clear_gpu_cache():
    """Clear GPU cache to free up memory
    
    This is important when running multiple model evaluations sequentially
    to avoid out-of-memory errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_random_architecture(min_layers=3, max_layers=8):
    """Generate a random neural network architecture
    
    Creates CNN architectures with random:
    - Depth (number of layers between min_layers and max_layers)
    - Channel counts (16, 32, 64, or 128 per layer)
    - Stride patterns (mostly 1s with some 2s)
    
    Args:
        min_layers (int): Minimum number of layers
        max_layers (int): Maximum number of layers
        
    Returns:
        tuple: (model, channels, strides) - the generated architecture and its configuration
    """
    num_classes = 10
    
    # Randomly select number of layers
    num_layers = random.randint(min_layers, max_layers)
    
    # Generate random channel configurations
    channels = [random.choice([16, 32, 64, 128]) for _ in range(num_layers)]
    
    # Generate random stride configurations (with more 2's than 1's)
    strides = []
    for i in range(num_layers):
        # More likely to have stride 2 than stride 1
        stride = 2 if random.random() < 0.9 else 1
        strides.append(stride)
    
    # Clear GPU cache before creating a new model
    clear_gpu_cache()
    
    # Create model with random architecture using the imported NeuralNetwork
    model = NeuralNetwork(num_classes, channels, strides)
    
    return model, channels, strides

def run_neural_architecture_search(data_loader, num_iterations=20):
    """Run a simple NAS using activation covariance as an evaluation metric
    
    This function:
    1. Generates random architectures in each iteration
    2. Evaluates each architecture using the activation score metric
    3. Tracks the best performing architecture and all results
    4. Returns the best model and all evaluation results
    
    Args:
        data_loader: DataLoader for evaluation data
        num_iterations: Number of architectures to evaluate
        
    Returns:
        tuple: (best_model, results) - best architecture found and all results
    """
    best_score = -float('inf')
    best_architecture = None
    best_channels = None
    best_strides = None
    
    results = []
    
    for i in tqdm(range(num_iterations), desc="NAS Progress"):
        # Generate random architecture
        model, channels, strides = generate_random_architecture()
        
        # Calculate activation score
        score = calculate_activation_score(model, data_loader, num_batches=5)
        
        # Clear GPU cache after evaluation
        clear_gpu_cache()
        
        # Keep track of results
        results.append({
            'channels': channels,
            'strides': strides,
            'score': score
        })
        
        # Update best architecture
        if score > best_score:
            best_score = score
            best_architecture = copy.deepcopy(model)
            best_channels = channels.copy()
            best_strides = strides.copy()
            print(f"New best architecture found! Score: {best_score:.4f}")
            print(f"Channels: {best_channels}")
            print(f"Strides: {best_strides}")
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return best_architecture, results

def main():
    """
    Entry point that:
    1. Prepares a small CIFAR10 subset for efficient architecture evaluation
    2. Runs the neural architecture search process
    3. Displays the top performing architectures
    4. Saves the best model found
    
    This function uses a small subset of the CIFAR10 dataset and performs NAS to find
    a well-performing CNN architecture based on activation diversity.
    """
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use a small subset of CIFAR10 for quick evaluation
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    subset_size = 1000
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, subset_indices)
    data_loader = DataLoader(subset, batch_size=64, shuffle=False)
    
    print("Starting Neural Architecture Search...")
    best_model, results = run_neural_architecture_search(data_loader, num_iterations=20)
    
    # Print top 3 architectures
    print("\nTop 3 architectures:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Channels: {result['channels']}")
        print(f"   Strides: {result['strides']}")
    
    # Save the best model
    torch.save(best_model.state_dict(), 'nas_best_model.pth')
    print("\nBest model saved to nas_best_model.pth")

if __name__ == "__main__":
    main()
