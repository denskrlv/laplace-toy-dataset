import matplotlib.pyplot as plt


def plot_results(results, suite_type, device):
    """
    Generates and saves a plot of the timing results.
    """
    header_map = {'N': 'N_Samples', 'M': 'N_Features', 'C': 'N_Classes'}
    x_label = header_map[suite_type]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in results.items():
        # Unpack the data: levels are the x-axis, times are the y-axis
        levels = [item[0] for item in data]
        times = [item[1] for item in data]
        
        ax.plot(levels, times, marker='o', linestyle='-', label=model_name.upper())

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Wall-clock Time (seconds)", fontsize=12)
    ax.set_title(f"Model Scaling Performance vs. {x_label} (on {device.upper()})", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log') # Use a log scale for time, as it can vary greatly
    ax.grid(True, which="both", ls="--")

    # Save the plot to a file
    filename = f"scaling_results_{suite_type}.png"
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")


import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def visualize_cifar10_examples(n_examples=10):
    """
    Loads the CIFAR-10 dataset and displays a grid of example images.
    This represents the data used in the `Scale-N` suite.
    """
    # Load the CIFAR-10 dataset
    # We download it if it's not already in the './data' directory
    cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    
    # CIFAR-10 class names for labeling the plot
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create a figure to display the images
    fig, axes = plt.subplots(1, n_examples, figsize=(15, 3))
    fig.suptitle('Examples from the CIFAR-10 Dataset (Scale-N)', fontsize=16)

    for i in range(n_examples):
        # Get a random image and its label
        idx = np.random.randint(0, len(cifar_dataset))
        img, label = cifar_dataset[idx]
        
        # The image is a Tensor of shape (3, 32, 32). We need to change it to
        # (32, 32, 3) for matplotlib to display it correctly.
        img_display = img.permute(1, 2, 0).numpy()
        
        ax = axes[i]
        ax.imshow(img_display)
        ax.set_title(classes[label])
        ax.axis('off') # Hide the axes ticks

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for title
    # Save the figure to a file
    plt.savefig("cifar10_examples.png")
    print("Saved CIFAR-10 examples plot to cifar10_examples.png")
    plt.show()


def visualize_make_blobs_example(n_features=2):
    """
    Generates synthetic data using make_blobs and displays a scatter plot.
    This represents the data used in the `Scale-M` suite.
    Note: We can only visualize 2D data, but this serves as a great illustration.
    """
    # Generate 2D data for visualization purposes
    X, y = make_blobs(n_samples=500, centers=4, n_features=n_features, random_state=42, cluster_std=1.5)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50, alpha=0.8)
    
    plt.title(f'Example of 2D Synthetic Data (Scale-M)', fontsize=16)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'Class {i}' for i in range(4)])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add a note explaining this is a 2D illustration of a higher-dimensional dataset
    plt.figtext(0.5, 0.01, 'Note: This is a 2D visualization. The `Scale-M` suite uses up to 4096 features.', 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Save the figure to a file
    plt.savefig("make_blobs_example.png")
    print("Saved make_blobs example plot to make_blobs_example.png")
    plt.show()


if __name__ == '__main__':
    print("--- Generating CIFAR-10 Visualization ---")
    visualize_cifar10_examples()
    
    print("\n--- Generating make_blobs Visualization ---")
    visualize_make_blobs_example()
