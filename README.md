# H&M Fashion Recommendations

A personalized fashion recommendation system using deep learning, built with PyTorch Lightning and modern MLOps practices.

## Project Overview

This project implements a recommendation system for H&M fashion products using:
- **Collaborative filtering** - Matrix factorization with learned embeddings
- **Content-based filtering** - Transformer-based sequential recommendations
- **Hybrid approach** - Gated combination of both methods

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS credentials configured for S3 access (for DVC remote)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fashion-recommendations.git
cd fashion-recommendations

# Install dependencies
uv sync

# Install fish shell completions (optional)
cp completions/fashionctl.fish ~/.config/fish/completions/

# Configure W&B (first time only)
wandb login
```

### S3 Configuration for DVC

Configure AWS credentials for accessing the DVC remote:

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Option 2: AWS CLI configuration
aws configure
```

## Train

### 1. Download Data

```bash
# Download all datasets
fashionctl datasets download --all

# Or download specific datasets
fashionctl datasets download articles customers transactions

# List available datasets
fashionctl datasets list
```

### 2. Train Model

```bash
# Train with default configuration (hybrid model)
fashionctl models train

# Train with specific model type
fashionctl models train model=collaborative
fashionctl models train model=content

# Train with custom overrides
fashionctl models train training.max_epochs=100 model.embedding_dim=256

# Debug mode (quick training for testing)
fashionctl models train training=debug

# Resume from checkpoint
fashionctl models train --resume checkpoints/last.ckpt

# Dry run (show config without training)
fashionctl models train --dry-run
```

### 3. Run Inference

```bash
# Inference with best checkpoint
fashionctl models infer

# Inference with specific checkpoint
fashionctl models infer --checkpoint checkpoints/epoch=5-val_loss=0.1234.ckpt

# Custom input/output paths
fashionctl models infer --input-path data/test_customers.csv --output-path results.csv
```

### 4. Export Model

```bash
# Export to ONNX format
fashionctl models export --checkpoint checkpoints/last.ckpt --output model.onnx
```

## Configuration

Configurations are managed via Hydra and located in `configs/`:

```
configs/
  train.yaml          # Main training config
  infer.yaml          # Inference config
  model/
    hybrid.yaml       # Hybrid model (default)
    collaborative.yaml
    content.yaml
  data/
    default.yaml      # Training data config
    inference.yaml    # Inference data config
  training/
    default.yaml      # Training hyperparameters
    debug.yaml        # Quick debug training
  wandb/
    default.yaml      # W&B logging settings
```

Override any config value via CLI:

```bash
fashionctl models train model.learning_rate=0.0001 training.max_epochs=100
```

## Project Structure

```
fashion-recommendations/
  src/fashion/
    cli.py            # CLI entry point
    console.py        # Rich console utilities
    commands/
      datasets.py     # Dataset management commands
      models.py       # Model management commands
    data/
      dataset.py      # PyTorch datasets
      datamodule.py   # Lightning data module
    models/
      architectures.py  # Neural network architectures
      lightning.py      # Lightning module
  configs/            # Hydra configurations
  data/               # Data directory (DVC tracked)
  checkpoints/        # Model checkpoints
  logs/               # Training logs
  completions/        # Shell completions
```

## Remote Server Workflow

```bash
# On remote server
git clone https://github.com/your-username/fashion-recommendations.git
cd fashion-recommendations

# Install dependencies
uv sync

# Download data
fashionctl datasets download --all

# Start training
fashionctl models train

# Run inference
fashionctl models infer
```

## Experiment Tracking

All experiments are logged to Weights & Biases, including:
- Training/validation loss
- Accuracy, precision, recall metrics
- Learning rate schedules
- Model hyperparameters
- Git commit hash (and jj change-id if using Jujutsu)

View experiments at: https://wandb.ai/your-entity/fashion-recommendations
