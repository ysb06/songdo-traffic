# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a traffic prediction project (`traffic_imc`) that implements and validates machine learning models for traffic flow forecasting using the METR-IMC dataset. The project focuses on comparing different neural network architectures, particularly LSTM-based RNN models and Spatio-Temporal Graph Convolutional Networks (STGCN).

## Architecture

### Package Structure
- `src/metr_val/`: Main package containing model implementations and utilities
  - `models/`: Neural network model implementations
    - `rnn.py`: LSTM-based models with PyTorch Lightning wrapper (`LSTMLightningModule`, `LSTMBaseModel`)
    - `stgcn/`: Spatio-Temporal Graph Convolutional Network implementation (copied from third-party)
  - `__main__.py`: Main training script that trains RNN models using Lightning Trainer
  - `test_performance.py`: Performance evaluation utilities for RNN models
  - `utils.py`: Configuration utilities including `PathConfig` for YAML-based path management

### Data Flow
1. Traffic data is loaded from HDF5 format (`data/selected_small_v1/metr-imc.h5`)
2. Data preprocessing handled by external `metr.datasets` package dependency
3. Models trained using PyTorch Lightning with WandB logging
4. Results saved to `output/` directory with model checkpoints

### Key Dependencies
- External package: `songdo-metr` (local development dependency for data handling)
- PyTorch Lightning for training infrastructure
- WandB for experiment tracking and logging
- Third-party STGCN implementation in `../third_party/stgcn/model/`

## Development Commands

### Environment Setup
```bash
# Install dependencies with PDM
pdm install

# Copy STGCN model files from third-party (run after install)
pdm run copy-stgcn
# or manually:
python scripts/post_update.py
```

### Training and Evaluation
```bash
# Run main RNN training
python -m src.metr_val

# Run performance testing
python -m src.metr_val.test_performance --data_dir ../datasets/metr-imc --results_dir ./rnn_results
```

### Third-Party Integration
The project uses a post-install script (`scripts/post_update.py`) that automatically copies STGCN model files from `../third_party/stgcn/model/` to `src/metr_val/models/stgcn/` and fixes import statements for proper relative imports.

## Key Implementation Details

### Model Architecture
- **RNN Models**: LSTM-based with configurable hidden size, layers, and dropout
- **STGCN Models**: Two variants using different graph convolution approaches (Chebyshev polynomials vs standard GraphConv)
- Both models support PyTorch Lightning training with automatic metric logging

### Data Handling
- Uses external `metr.datasets.rnn.datamodule.SimpleTrafficDataModule` for data loading
- Supports scaling/normalization with sklearn MinMaxScaler
- Metrics calculated on both scaled and original data ranges

### Experiment Tracking
- WandB integration for experiment logging (project: "Traffic-IMC")
- Model checkpointing with early stopping based on validation loss
- CSV logging available as alternative to WandB

## Important Notes

- The project depends on a local `songdo-metr` package that must be available in the parent directory
- STGCN models require manual copying from third-party sources via the post-install script
- Training outputs include both scaled metrics (for model comparison) and original scale metrics (for real-world interpretation)
- All model training uses automatic device detection (GPU if available)