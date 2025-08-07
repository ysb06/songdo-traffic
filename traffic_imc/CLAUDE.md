# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a traffic prediction project using PyTorch Lightning and LSTM models for time series forecasting on the Songdo traffic dataset. The project structure follows a Python package layout with PDM for dependency management.

## Development Commands

### Environment Setup
```bash
# Install dependencies with PDM
pdm install

# Activate virtual environment  
pdm shell
# or use: source .venv/bin/activate
```

### Running the Model
```bash
# Run training (main entry point)
python -m metr_val

# Run performance testing
python -m metr_val.test_performance

# With custom parameters
python -m metr_val.test_performance --seq_length 48 --hidden_size 128 --max_epochs 50
```

### Data Requirements
- Traffic data should be placed in `./data/selected_small/metr-imc.h5`
- The project expects HDF5 format data files
- Additional supporting files: `adj_mx.pkl`, `distances_imc.csv`, `graph_sensor_locations.csv`, `metadata.h5`

## Code Architecture

### Core Components

**Main Training Module** (`src/metr_val/__main__.py`):
- Entry point for model training
- Configures PyTorch Lightning Trainer with Weights & Biases logging
- Uses TrafficDataModule from external `metr` package for data loading

**LSTM Model** (`src/metr_val/models/rnn.py`):
- `LSTMLightningModule`: Lightning wrapper with training/validation/test steps
- `LSTMBaseModel`: Core PyTorch LSTM implementation
- Supports both scaled and original scale metrics (MAE, RMSE, MAPE)
- Built-in MinMaxScaler integration for data normalization

**Performance Testing** (`src/metr_val/test_performance.py`):
- Comprehensive model evaluation framework
- Command-line interface for hyperparameter tuning
- Automated CSV data loading and Lightning integration
- Results logging and model checkpointing

**Utilities** (`src/metr_val/utils.py`):
- `PathConfig`: YAML-based configuration management
- Support for nested path configurations with attribute and dictionary access

### External Dependencies
- Depends on external `songdo-metr` package (installed as editable dependency)
- Uses `metr.datasets.rnn.datamodule.TrafficDataModule` for data handling
- Weights & Biases integration for experiment tracking

### Model Configuration
- Default sequence length: 24 timesteps
- LSTM hidden size: 64 units
- Number of layers: 2
- Early stopping with validation loss monitoring
- Learning rate scheduling with ReduceLROnPlateau

### Output Structure
- Model checkpoints saved to `./output/rnn/`
- Weights & Biases logs project: "Traffic-IMC"
- Performance test results in configurable results directory