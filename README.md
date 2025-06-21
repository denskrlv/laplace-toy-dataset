# Laplace Toy Dataset

A toy dataset and benchmarking suite for testing the computational efficiency of Bayesian Deep Learning methods, specifically comparing Laplace Approximation (LA), Deep Ensembles (DE), and MAP estimation.

## Overview

This project provides a controlled environment to measure how different Bayesian inference techniques scale with:

1. **Data Size (Scale-N)**: Using subsets of CIFAR-10 with varying number of samples
2. **Feature Dimensionality (Scale-M)**: Using synthetic data with varying feature dimensions

The primary goal is to validate claims about the computational efficiency of different Bayesian approaches, particularly those made in the paper ["Laplace Redux – Effortless Bayesian Deep Learning"](https://arxiv.org/abs/2106.14806) by Daxberger et al. (2021).

## Installation

1. Clone the repository:

```python
git clone https://github.com/yourusername/laplace-toy-dataset.git
cd laplace-toy-dataset
```

2. Create and activate a virtual environment (optional but recommended):

```python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:

```python
pip install -r requirements.txt
```

4. Install the package in development mode:

```python
pip install -e .
```

## Usage

### Running Experiments

To run experiments, use the experiment.py script with appropriate arguments:

```python
# Run Scale-N experiment (varying number of samples)
python experiment.py --suite N --model_to_test all --device auto

# Run Scale-M experiment (varying number of features)
python experiment.py --suite M --model_to_test all --device auto
```

### Arguments:

--suite: Type of scaling experiment (N for samples, M for features)
--model_to_test: Which models to evaluate (map, la, de, or all)
--device: Computing device (cpu, cuda, mps, or auto for automatic selection)

### Visualizing Data

To generate visualizations of the datasets used:

```python
python visualizer.py
```

This will create example visualizations of both the CIFAR-10 samples and the synthetic data in the images directory.

## Repository Structure

```python
.
├── data/                   # Dataset directory (automatically populated)
├── images/                 # Visualization images
├── results/                # Experiment result plots
├── dataset.py              # Dataset generation utilities
├── experiment.py           # Main experiment code
├── visualizer.py           # Visualization utilities
├── setup.py                # Package setup file
├── requirements.txt        # Project dependencies
├── BLOG.md                 # Blog post explaining the project
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Results

After running experiments, results will be:

1. Displayed in the console
2. Saved as PNG plots in the project root directory (e.g., scaling_results_N.png)

Example results can be found in the results directory.
