import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR10 # CIFAR100 import removed
from torchvision.transforms import ToTensor
from sklearn.datasets import make_blobs
from laplace import Laplace

import time
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# --- Model Definitions, Dataset Logic, and Training Logic ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_features, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)

def get_scale_n_dataset(n_samples):
    full_train_set = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    return Subset(full_train_set, list(range(min(n_samples, len(full_train_set)))))

def get_scale_m_dataset(n_features, n_samples=5000, n_classes=10):
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=42, cluster_std=2.0)
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())

# The get_scale_c_dataset function has been completely removed.

def train_map_model(model, train_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
    return model

def run_single_test(model_class, model_args, train_loader, device, config):
    results = {}
    get_mem_str = lambda mem_val: f", Peak Mem={mem_val:.2f}MB" if device == 'cuda' else ""
    if config['model_to_test'] in ['map', 'all']:
        if device == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model_map = model_class(**model_args).to(device)
        start_time = time.monotonic()
        train_map_model(model_map, train_loader, device)
        time_taken = time.monotonic() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device == 'cuda' else 0
        results['map'] = (time_taken, peak_mem)
        print(f"MAP: Time={time_taken:.2f}s{get_mem_str(peak_mem)}")
    if config['model_to_test'] in ['la', 'all']:
        if device == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model_la = model_class(**model_args).to(device)
        start_time = time.monotonic()
        trained_model_la = train_map_model(model_la, train_loader, device)
        la = Laplace(trained_model_la, 'classification', subset_of_weights='last_layer', hessian_structure='kron')
        la.fit(train_loader)
        time_taken = time.monotonic() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device == 'cuda' else 0
        results['la'] = (time_taken, peak_mem)
        print(f"LA:  Time={time_taken:.2f}s{get_mem_str(peak_mem)}")
    if config['model_to_test'] in ['de', 'all']:
        N_ENSEMBLES = 5
        if device == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        start_time = time.monotonic()
        for _ in range(N_ENSEMBLES):
            model_de = model_class(**model_args).to(device)
            train_map_model(model_de, train_loader, device)
        time_taken = time.monotonic() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device == 'cuda' else 0
        results['de'] = (time_taken, peak_mem)
        print(f"DE:  Time={time_taken:.2f}s{get_mem_str(peak_mem)}")
    return results

def run_suite(suite_type, levels, model_class, get_dataset_fn, get_model_args_fn, config, device):
    all_results = defaultdict(list)
    for level in levels:
        print(f"\n--- Testing with {suite_type.replace('_', ' ')} = {level} ---")
        train_dataset = get_dataset_fn(level)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        model_args = get_model_args_fn(level)
        level_results = run_single_test(model_class, model_args, train_loader, device, config)
        for model_name, res in level_results.items():
            all_results[model_name].append((level, res[0], res[1]))
    return all_results

def print_results(results, suite_type, device):
    header_map = {'N': 'N_Samples', 'M': 'N_Features'} # 'C' removed
    header = header_map[suite_type]
    print("\n\n--- Final Results ---")
    for model_name, data in results.items():
        print(f"\nModel: {model_name.upper()}")
        mem_header = "| Peak Memory (MB)" if device == 'cuda' else ""
        print(f"{header:<12} | Time (s) {mem_header}")
        print("-" * (25 + len(mem_header)))
        for level, time_taken, peak_mem in data:
            mem_str = f"| {peak_mem:<16.2f}" if device == 'cuda' else ""
            print(f"{level:<12} | {time_taken:<8.2f} {mem_str}")

def plot_results(results, suite_type, device):
    header_map = {'N': 'N_Samples', 'M': 'N_Features'} # 'C' removed
    x_label = header_map[suite_type]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, data in results.items():
        if not data: continue
        levels = [item[0] for item in data]
        times = [item[1] for item in data]
        ax.plot(levels, times, marker='o', linestyle='-', label=model_name.upper())
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Wall-clock Time (seconds)", fontsize=12)
    ax.set_title(f"Model Scaling Performance vs. {x_label} (on {device.upper()})", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--")
    filename = f"scaling_results_{suite_type}.png"
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run scaling experiments.")
    parser.add_argument('--suite', type=str, choices=['N', 'M'], default='N', 
                        help='Which scaling suite to test: N (samples) or M (features).') # 'C' removed from help text
    parser.add_argument('--model_to_test', type=str, choices=['map', 'la', 'de', 'all'], default='all', 
                        help='Which model to test.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps', 'auto'], default='auto',
                        help='Device to run the experiment on. Default is "auto".')
    args = parser.parse_args()
    
    # Device Selection Logic
    if args.device == 'auto':
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available():
            print("Warning: MPS backend is selected. Library compatibility is not guaranteed. CPU is recommended.")
            device = "mps"
        else: device = "cpu"
    else:
        if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError("CUDA not available.")
        if args.device == 'mps' and not torch.backends.mps.is_available(): raise ValueError("MPS not available.")
        device = args.device
    print(f"Running experiment on device: {device}")

    config = vars(args)
    
    if args.suite == 'N':
        final_results = run_suite('N_Samples', [1000, 5000, 10000, 20000], SimpleCNN, 
                                  get_scale_n_dataset, lambda level: {'num_classes': 10}, config, device)
    elif args.suite == 'M':
        final_results = run_suite('N_Features', [64, 256, 1024, 4096], SimpleMLP,
                                  get_scale_m_dataset, lambda level: {'input_features': level, 'num_classes': 10}, config, device)
    
    print_results(final_results, args.suite, device)
    plot_results(final_results, args.suite, device)
