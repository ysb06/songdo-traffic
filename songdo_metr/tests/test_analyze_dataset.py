#!/usr/bin/env python3
"""
Script to analyze the results of generate_raw_dataset function
"""
import os
import pandas as pd
import pickle
from datetime import datetime
from metr.components import TrafficData, Metadata, SensorLocations, DistancesImc, IdList, AdjacencyMatrix
from metr.utils import PathConfig

def print_file_info(file_path: str, file_type: str):
    """Print basic file information"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"✓ {file_type}: {os.path.basename(file_path)} ({size:.2f} MB)")
        return True
    else:
        print(f"✗ {file_type}: {os.path.basename(file_path)} (Not found)")
        return False

def analyze_traffic_data(file_path: str):
    """Analyze the main traffic data file"""
    print("\n" + "="*60)
    print("TRAFFIC DATA ANALYSIS")
    print("="*60)
    
    traffic_data = TrafficData.import_from_hdf(file_path)
    df = traffic_data.data
    
    print(f"Data Shape: {df.shape}")
    print(f"Time Range: {df.index.min()} to {df.index.max()}")
    print(f"Number of Sensors: {len(df.columns)}")
    print(f"Total Data Points: {df.size:,}")
    
    # NaN analysis
    total_values = df.size
    nan_values = df.isna().sum().sum()
    nan_percentage = (nan_values / total_values) * 100
    
    print(f"\nMissing Data Analysis:")
    print(f"  Total NaN values: {nan_values:,}")
    print(f"  NaN percentage: {nan_percentage:.2f}%")
    
    # Per-sensor NaN analysis
    sensor_nan_stats = df.isna().mean() * 100
    print(f"  Sensors with 0% NaN: {(sensor_nan_stats == 0).sum()}")
    print(f"  Sensors with <10% NaN: {(sensor_nan_stats < 10).sum()}")
    print(f"  Sensors with 50%+ NaN: {(sensor_nan_stats >= 50).sum()}")
    print(f"  Average NaN per sensor: {sensor_nan_stats.mean():.2f}%")
    
    # Data distribution
    print(f"\nData Statistics:")
    all_values = df.values.flatten()
    all_values = all_values[~pd.isna(all_values)]
    print(f"  Min value: {all_values.min():.2f}")
    print(f"  Max value: {all_values.max():.2f}")
    print(f"  Mean value: {all_values.mean():.2f}")
    print(f"  Median value: {pd.Series(all_values).median():.2f}")
    
    return df

def analyze_metadata(file_path: str):
    """Analyze metadata file"""
    print("\n" + "="*60)
    print("METADATA ANALYSIS")
    print("="*60)
    
    metadata = Metadata.import_from_hdf(file_path)
    df = metadata.data
    
    print(f"Metadata Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'MAX_SPD' in df.columns:
        print(f"\nSpeed Limits:")
        print(f"  Range: {df['MAX_SPD'].min()}-{df['MAX_SPD'].max()} km/h")
        print(f"  Most common: {df['MAX_SPD'].mode().iloc[0]} km/h")
    
    if 'LANES' in df.columns:
        print(f"\nLane Information:")
        print(f"  Range: {df['LANES'].min()}-{df['LANES'].max()} lanes")
        print(f"  Most common: {df['LANES'].mode().iloc[0]} lanes")
        
    return df

def analyze_distances(file_path: str):
    """Analyze distance data"""
    print("\n" + "="*60)
    print("DISTANCE DATA ANALYSIS")
    print("="*60)
    
    distances = DistancesImc.import_from_csv(file_path)
    df = distances.data
    
    print(f"Distance Matrix Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Use correct column names
    from_col = 'from' if 'from' in df.columns else 'from_id'
    to_col = 'to' if 'to' in df.columns else 'to_id'
    
    print(f"Unique sensors: {len(set(df[from_col].unique()) | set(df[to_col].unique()))}")
    print(f"Total connections: {len(df)}")
    
    print(f"\nDistance Statistics:")
    print(f"  Min distance: {df['distance'].min():.2f} meters")
    print(f"  Max distance: {df['distance'].max():.2f} meters")
    print(f"  Mean distance: {df['distance'].mean():.2f} meters")
    print(f"  Median distance: {df['distance'].median():.2f} meters")
    
    return df

def analyze_adjacency_matrix(file_path: str):
    """Analyze adjacency matrix"""
    print("\n" + "="*60)
    print("ADJACENCY MATRIX ANALYSIS")
    print("="*60)
    
    with open(file_path, 'rb') as f:
        adj_data = pickle.load(f)
    
    print(f"Adjacency Data Type: {type(adj_data)}")
    
    if isinstance(adj_data, tuple):
        print(f"Tuple Length: {len(adj_data)}")
        sensor_ids, sensor_id_to_ind, adj_mx = adj_data
        print(f"Sensor IDs: {len(sensor_ids)} sensors")
        print(f"ID mapping: {len(sensor_id_to_ind)} entries")
        print(f"Adjacency Matrix Shape: {adj_mx.shape}")
        print(f"Matrix Type: {type(adj_mx)}")
        print(f"Non-zero connections: {(adj_mx > 0).sum()}")
        print(f"Sparsity: {((adj_mx == 0).sum() / adj_mx.size * 100):.2f}% zeros")
        return adj_mx
    else:
        adj_mx = adj_data
        print(f"Adjacency Matrix Shape: {adj_mx.shape}")
        print(f"Matrix Type: {type(adj_mx)}")
        print(f"Non-zero connections: {(adj_mx > 0).sum()}")
        print(f"Sparsity: {((adj_mx == 0).sum() / adj_mx.size * 100):.2f}% zeros")
        return adj_mx

def test_analyze_dataset():
    # Load configuration
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    print("SONGDO TRAFFIC DATASET ANALYSIS")
    print("="*60)
    print("Generated by generate_raw_dataset() function")
    print("="*60)
    
    # Check file existence
    print("\nFILE EXISTENCE CHECK:")
    files_info = [
        (PATH_CONF.metr_imc_path, "Traffic Data"),
        (PATH_CONF.metadata_path, "Metadata"),
        (PATH_CONF.sensor_ids_path, "Sensor IDs"),
        (PATH_CONF.sensor_locations_path, "Sensor Locations"),
        (PATH_CONF.distances_path, "Distance Data"),
        (PATH_CONF.adj_mx_path, "Adjacency Matrix"),
        (PATH_CONF.nodelink_node_path, "Node Data"),
        (PATH_CONF.nodelink_link_path, "Link Data"),
        (PATH_CONF.nodelink_turn_path, "Turn Data"),
        (PATH_CONF.imcrts_path, "IMCRTS Raw Data"),
    ]
    
    existing_files = []
    for file_path, file_type in files_info:
        if print_file_info(file_path, file_type):
            existing_files.append((file_path, file_type))
    
    # Detailed analysis
    try:
        # Analyze main traffic data
        traffic_df = analyze_traffic_data(PATH_CONF.metr_imc_path)
        
        # Analyze metadata
        metadata_df = analyze_metadata(PATH_CONF.metadata_path)
        
        # Analyze distances
        distances_df = analyze_distances(PATH_CONF.distances_path)
        
        # Analyze adjacency matrix
        adj_mx = analyze_adjacency_matrix(PATH_CONF.adj_mx_path)
        
        # Cross-validation
        print("\n" + "="*60)
        print("CROSS-VALIDATION")
        print("="*60)
        
        print(f"Traffic data sensors: {len(traffic_df.columns)}")
        print(f"Metadata entries: {len(metadata_df)}")
        
        # Handle adjacency matrix properly
        if isinstance(adj_mx, tuple):
            adj_matrix_shape = adj_mx[2].shape if len(adj_mx) > 2 else "Unknown"
            adj_size = adj_mx[2].shape[0] if len(adj_mx) > 2 and hasattr(adj_mx[2], 'shape') else "Unknown"
        else:
            adj_matrix_shape = adj_mx.shape
            adj_size = adj_mx.shape[0] if hasattr(adj_mx, 'shape') else "Unknown"
            
        print(f"Adjacency matrix size: {adj_matrix_shape}")
        
        # Check consistency
        if len(traffic_df.columns) == len(metadata_df) == adj_size:
            print("✓ All components have consistent sensor counts")
        else:
            print("⚠ Inconsistent sensor counts between components")
            print(f"  Traffic: {len(traffic_df.columns)}, Metadata: {len(metadata_df)}, Adjacency: {adj_size}")
            
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
