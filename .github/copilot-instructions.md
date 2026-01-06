# Copilot Instructions for songdo-traffic

## Project Overview

Multi-workspace traffic prediction project focused on Incheon Metropolitan City (송도/Songdo), Korea. Collects Korean government traffic data, processes road network data, and trains spatiotemporal ML models (STGCN, MLCAFormer, RNN) for traffic volume prediction using METR-LA compatible datasets.

## Workspace Structure

This is a **VS Code multi-root workspace** with independent sub-projects:

- **songdo_metr/** - Core data pipeline: Downloads Korean node-link + traffic data from data.go.kr API, generates HDF5 datasets, adjacency matrices, and PyTorch Lightning data modules
- **traffic_imc/** - ML model experiments: Tests various traffic prediction models (MLCAFormer, STGCN, RNN) with WandB logging
- **project_om/** - Outliers & missing data: Preprocessing pipeline for outlier detection, missing value interpolation, and data quality analysis
- **notebooks/** - Jupyter analysis notebooks for EDA and visualization
- **third_party/** - External model implementations (DCRNN, MLCAFormer, STGCN) referenced by other projects

Each sub-project has its own `.venv`, `pyproject.toml`, and dependencies managed by **PDM**.

## Dataset Pipeline (songdo_metr)

**songdo_metr** is the core data processing pipeline that transforms raw Korean government traffic data into ML-ready datasets compatible with METR-LA format for spatiotemporal traffic prediction.

### Raw Data Sources

#### 1. IMCRTS Traffic Data (Incheon Metropolitan City Real-Time System)
- **Source**: Korean Government Open Data Portal (`data.go.kr`)
- **API**: `ICRoadVolStat/NodeLink_Trfc_DD` (daily aggregated traffic volumes)
- **Format**: JSON responses with hourly traffic counts per road link
- **Raw Structure**:
  ```python
  {
    "statDate": "2023-01-01",     # Date
    "linkID": "1610010300",        # Road link ID (Korean Standard Node-Link)
    "hour00": 120,                 # Traffic volume at 00:00-01:00
    "hour01": 85,                  # Traffic volume at 01:00-02:00
    ...
    "hour23": 150                  # Traffic volume at 23:00-24:00
  }
  ```
- **Collection Period**: Configurable (default: `IMCRTS_START_DATE` to `IMCRTS_END_DATE` in pipeline.py)
- **Target Region**: Incheon Metropolitan City (region codes: 161-169)
- **Intermediate Output**: Pickle file (`imcrts_data.pkl`) containing raw DataFrame

#### 2. Korean Standard Node-Link Data
- **Source**: Korean ITS Portal (`its.go.kr`)
- **Format**: Shapefile (SHP) with road network geometry and metadata
- **Components**:
  - **NODE**: Road intersections with geographic coordinates
  - **LINK**: Road segments connecting nodes (includes LINK_ID, road type, capacity, length, geometry)
  - **TURN**: Allowed turning movements at intersections
- **Filter**: Only roads within Incheon Metropolitan City (region codes 161-169)
- **Intermediate Output**: Shapefiles (`nodelink_node.shp`, `nodelink_link.shp`, `nodelink_turn.shp`)

### Data Transformation Pipeline

#### Step 1: Raw Data Collection (`generate_raw_dataset()`)
```python
# In songdo_metr/src/metr/pipeline.py
generate_nodelink_raw()   # Downloads and filters Korean road network data
generate_imcrts_raw()      # Collects traffic data from data.go.kr API
generate_metr_imc_raw()    # Matches traffic data to road network
generate_dataset()         # Generates ML components
```

#### Step 2: Traffic Data Restructuring
**Process** (`TrafficData.import_from_pickle()`):
- Raw JSON (24 hourly columns per day) → Time-series DataFrame (hourly index)
- Pivot transformation:
  - **Before**: Each row = one day for one link (with hour00-hour23 columns)
  - **After**: Each row = one hour for all links (columns = LINK_IDs)
  
**Output Format**:
```python
# DataFrame structure
index: DatetimeIndex (hourly frequency)
columns: LINK_IDs (e.g., "1610010300", "1620023400", ...)
values: Traffic volumes (float, hourly vehicle counts)

# Example:
                    1610010300  1620023400  1630045600
2023-01-01 00:00:00      120.0       85.0      200.0
2023-01-01 01:00:00       95.0       70.0      180.0
2023-01-01 02:00:00       60.0       45.0      120.0
...
```

#### Step 3: Sensor Matching
**Process** (`generate_metr_imc_raw()`):
- Filters traffic data to include only LINK_IDs present in Korean Standard Node-Link data
- Ensures spatial consistency between traffic observations and road network geometry
- **Output**: `metr-imc.h5` (HDF5 file with time-series traffic data)

#### Step 4: Component Generation (`generate_dataset()`)
Generates five core ML components from matched data:

1. **Sensor IDs** (`sensor_ids.txt`):
   - Plain text list of all LINK_IDs in the dataset
   - Used for filtering and validation

2. **Metadata** (`metadata.h5`):
   - Road attributes per sensor: road type, lanes, length, capacity, speed limit
   - Extracted from Korean Standard Node-Link LINK table
   - HDF5 format for fast indexed access

3. **Sensor Locations** (`sensor_locations.csv`):
   - Geographic coordinates (longitude, latitude) for each LINK_ID
   - Computed as centroid of road segment geometry
   - CSV format: `sensor_id,longitude,latitude`

4. **Distances** (`distances.csv`):
   - Pairwise road network distances between all sensors (up to 9km limit)
   - Uses Korean Standard Node-Link TURN table for allowed movements
   - Computed via Dijkstra's algorithm on road network graph
   - CSV format: `from,to,cost` (cost = network distance in meters)

5. **Adjacency Matrix** (`adj_mx.pkl`):
   - Weighted spatial graph for GNN models
   - Formula: `W[i,j] = exp(-dist[i,j]²/σ²)` (Gaussian kernel with configurable σ)
   - Shape: `[num_sensors, num_sensors]`
   - Pickle format (NumPy array)

### Final Dataset Structure

**Output Directory** (configured in `config.yaml`):
```
datasets/metr-imc/
├── metr-imc.h5              # Time-series traffic data [time × sensors]
├── sensor_ids.txt           # List of LINK_IDs
├── metadata.h5              # Road attributes per sensor
├── sensor_locations.csv     # Geographic coordinates
├── distances.csv            # Network distances matrix (sparse)
├── adj_mx.pkl               # Adjacency matrix for GNN
├── metr-imc.shp             # Shapefile of roads with traffic data (visualization)
└── distances.shp            # Shapefile of distance edges (visualization)
```

**Dataset Characteristics**:
- **Temporal**: Hourly granularity, continuous time series (months to years)
- **Spatial**: 100-500+ road sensors (varies by filtering criteria)
- **Format**: METR-LA compatible (can be used with DCRNN, STGCN, etc.)
- **Data Quality**: Includes outlier detection and interpolation pipelines (via `project_om`)

### PyTorch Integration

Dataset classes in `songdo_metr/src/metr/datasets/` provide PyTorch Lightning DataModules:
- `MLCAFormerDataModule`: Multi-level attention models
- `DCRNNDataModule`: Diffusion convolutional RNN
- `STGCNDataModule`: Spatiotemporal graph convolution

**Features**:
- Automatic train/val/test splits by date range
- Sliding window generation (e.g., 12 steps in → 12 steps out)
- Built-in normalization (StandardScaler, MinMaxScaler)
- Temporal feature extraction (hour-of-day, day-of-week embeddings)

**Usage Example**:
```python
from metr.datasets import MLCAFormerDataModule

dm = MLCAFormerDataModule(
    dataset_path="../datasets/metr-imc/metr-imc.h5",
    training_period=("2023-01-01", "2023-12-31"),
    in_steps=12,   # Use past 12 hours
    out_steps=12,  # Predict next 12 hours
    batch_size=64
)
dm.setup()
train_loader = dm.train_dataloader()
```

## Development Commands

### Environment Setup (Required First)
```bash
# Always activate the sub-project's virtual environment
cd <subproject>  # songdo_metr, traffic_imc, project_om, or notebooks
source .venv/bin/activate  # Or let VS Code select it

# Install dependencies with PDM (preferred)
pdm install

# Alternative: pip install -e .
```

### Core Workflows

**Data Pipeline (songdo_metr)**
```bash
cd songdo_metr
export PYTHONPATH="./src"
export PDP_KEY="[API key from data.go.kr]"
python -m metr.dataset  # Runs generate_raw_dataset() in pipeline.py
pytest tests/           # Run component tests
```

**ML Training (traffic_imc)**
```bash
cd traffic_imc
source .venv/bin/activate
python -m metr_val      # Trains MLCAFormer with WandB logging
# Models logged to wandb/, outputs to output/
```

**Outlier Processing (project_om)**
```bash
cd project_om
source .venv/bin/activate
python -m ppom  # Runs outlier detection + interpolation pipeline
```

## Architecture Patterns

### Component-Based Data Processing (songdo_metr)
Each data type has a dedicated class in `src/metr/components/`:
- **TrafficData**: Time-series traffic volumes (HDF5 storage)
- **AdjacencyMatrix**: Spatial graph structure (pickle)
- **Metadata**: Sensor metadata (HDF5)
- **SensorLocations**: Geographic coordinates (CSV)

All components follow:
```python
class Component:
    @staticmethod
    def import_from_<format>(filepath) -> Component
    
    def to_<format>(filepath)
    def process(processor: Processor)  # Outliers, interpolation
```

### PyTorch Lightning Integration
ML datasets in `songdo_metr/src/metr/datasets/` provide:
- DataModules with train/val/test splits
- Automatic scaling (StandardScaler, MinMaxScaler)
- Time-based features (hour-of-day, day-of-week embeddings)

Example: `MLCAFormerDataModule(dataset_path, training_period, in_steps=12, out_steps=12)`

### Cross-Project Dependencies
- `traffic_imc` and `project_om` depend on `songdo_metr` via PDM:
  ```toml
  [tool.pdm.dev-dependencies]
  dev = ["-e file:///${PROJECT_ROOT}/../songdo_metr#egg=songdo-metr"]
  ```
- Models from `third_party/` are imported into respective `models/` directories

### Configuration-Driven Paths
All data paths centralized in `config.yaml` (root: `../datasets/metr-imc`):
```yaml
dataset:
  filenames:
    metr_imc: "metr-imc.h5"
    adjacency_matrix: "adj_mx.pkl"
```
Loaded via: `PathConfig.from_yaml("../config.yaml")`

## Critical Conventions

### Python Version & Dependencies
- **Python 3.11** (all projects locked to `requires-python = "==3.11.*"`)
- **PDM** for dependency management (not pip/poetry)
- **src layout**: Always set `PYTHONPATH="./src"` before running modules

### Data Formats
- **HDF5** (`.h5`): Time-series traffic data, metadata
- **Pickle** (`.pkl`): Adjacency matrices, IMCRTS raw data
- **CSV**: Sensor locations, distances
- **Shapefile** (`.shp`): Geographic data (geopandas)

### ML Training Standards
- **WandB** for experiment tracking (projects: `Traffic-IMC-MLCAFormer`, etc.)
- **PyTorch Lightning** for training loops
- Outputs to `output/` directory, logs to `wandb/` directory
- Device selection: MPS (Apple Silicon) > CUDA > CPU

### Data Quality Pipeline
Outlier processors in `songdo_metr/src/metr/components/metr_imc/outlier.py`:
- `RemovingWeirdZeroOutlierProcessor`: Removes suspicious zero values
- `TrafficCapacityAbsoluteOutlierProcessor`: Caps by road capacity
- `HourlyInSensorZscoreOutlierProcessor`: Z-score by hour/sensor

Interpolators in `components/metr_imc/interpolation.py`:
- `LinearInterpolator`, `SplineLinearInterpolator`
- `TimeMeanFillInterpolator`, `MonthlyMeanFillInterpolator`

### Region Codes
Incheon Metropolitan City regions: `161-169` (hardcoded in `pipeline.py`)

## Integration Points

- **Korean Government APIs**: `data.go.kr` (requires `PDP_KEY` env var)
- **Korean Standard Node-Link**: Road network data from `its.go.kr`
- **External Models**: Import from `third_party/` submodules (git submodules)

## Testing
- pytest only in `songdo_metr/` (component tests)
- No formal testing in `traffic_imc` or `project_om` (experiment-focused)

## Key Files
- [config.yaml](config.yaml) - Centralized data path configuration
- [songdo_metr/src/metr/pipeline.py](songdo_metr/src/metr/pipeline.py) - Main data generation pipeline
- [songdo_metr/CLAUDE.md](songdo_metr/CLAUDE.md) - Detailed songdo_metr documentation
- [traffic_imc/src/metr_val/__main__.py](traffic_imc/src/metr_val/__main__.py) - ML training entry point
