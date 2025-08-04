# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**songdo_metr** is a traffic data processing and machine learning project for Incheon Metropolitan City (송도/Songdo), Korea. It collects traffic data from Korean government APIs, processes Korean Standard Node-Link road network data, and generates METR-LA compatible datasets for traffic volume prediction research.

## Common Development Commands

```bash
# Environment setup
export PYTHONPATH="./src"
export PDP_KEY="[API Key from data.go.kr]"

# Installation (primary method)
pdm install

# Alternative installation 
pip install -e .

# Run main data processing pipeline
python -m metr.dataset

# Run tests
pytest tests/

# Check Python path issues
python -c "import sys; print('\n'.join(sys.path))"
```

## Architecture Overview

### Core Pipeline Flow
1. **Data Collection**: Download Korean node-link data + collect traffic data via `data.go.kr` API
2. **Data Processing**: Convert, filter, and match traffic data to road network using `src/metr/pipeline.py`
3. **Graph Generation**: Create adjacency matrices and sensor locations via `src/metr/components/`
4. **Dataset Creation**: Generate ML-ready time-series datasets compatible with METR-LA format
5. **ML Integration**: PyTorch Lightning data modules in `src/metr/dataloader.py`

### Key Components Architecture
- **Component Pattern**: Each data type (TrafficData, Metadata, AdjacencyMatrix) has dedicated classes with import/export methods in `src/metr/components/`
- **Modular Processing**: `src/metr/imcrts/` handles Incheon traffic data collection, `src/metr/nodelink/` processes Korean road network data
- **Configuration-Driven**: `config.yaml` centralizes all data paths and file locations
- **PyTorch Lightning Integration**: Structured ML training with proper data modules in `src/metr/datasets/`

### Data Sources and Formats
- **Input**: Korean government traffic data (data.go.kr API), Korean Standard Node-Link road network data
- **Output**: HDF5 files, CSV datasets, pickle files, adjacency matrices
- **Target Region**: Incheon Metropolitan City (region codes 161-169)

### File Structure Patterns
- **src/metr/**: Main source code following src layout
- **src/metr/components/**: Core data processing modules
- **src/metr/datasets/**: PyTorch dataset classes for ML training
- **config.yaml**: Path configuration (default root: `../datasets/metr-imc`)
- **pyproject.toml**: PDM-based dependency management (Python >=3.10)

### Key Dependencies
- **Data Processing**: pandas, geopandas, numpy, scipy, networkx
- **ML Framework**: PyTorch, PyTorch Lightning, scikit-learn  
- **Storage**: HDF5 (tables), pickle
- **Geospatial**: shapely for spatial operations

## Development Notes

### Environment Requirements
- Python >=3.10 required
- API key from `data.go.kr` required for data collection
- PDM preferred for dependency management, pip as fallback

### Data Pipeline Entry Points
- **Main execution**: `python -m metr.dataset` runs `generate_raw_dataset()` from `pipeline.py`
- **Component testing**: Individual components in `src/metr/components/` can be imported and tested separately
- **ML training**: Use data modules from `src/metr/dataloader.py` with PyTorch Lightning

### Configuration Management
- **Paths**: Modify `config.yaml` to change data file locations
- **API Access**: Set `PDP_KEY` environment variable for data collection
- **Python Path**: Ensure `./src` is in PYTHONPATH for proper imports

## Note
- In this project, the virtual environment .venv is used. Always use this virtual environment when running the code.