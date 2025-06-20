import torch
from sklearn.datasets import make_blobs


def generate_scale_m_dataset(n_samples=10000, n_features=256, n_classes=10):
    """
    Generates a synthetic dataset for testing scaling with input dimensions.
    """
    # Generate data using scikit-learn
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_features,
        random_state=42,
        cluster_std=2.0
    )

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return dataset

# Example usage to generate one part of the suite
scale_m_256 = generate_scale_m_dataset(n_features=256)
print(f"Generated dataset with {len(scale_m_256)} samples and {scale_m_256[0][0].shape[0]} features.")
