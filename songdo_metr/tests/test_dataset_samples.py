#!/usr/bin/env python3
"""
Test script to display actual value samples from the dataset
"""
import os
import sys
import pandas as pd
import numpy as np
from metr.components import TrafficData, Metadata, SensorLocations, DistancesImc
from metr.utils import PathConfig

def display_traffic_data_samples():
    """Display head(5) samples from traffic data"""
    print("="*60)
    print("TRAFFIC DATA - ACTUAL VALUE SAMPLES")
    print("="*60)
    
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    if not os.path.exists(PATH_CONF.metr_imc_path):
        print(f"❌ Traffic data file not found: {PATH_CONF.metr_imc_path}")
        return None
    
    # Load traffic data
    traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
    df = traffic_data.data
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns (first 10): {list(df.columns[:10])}")
    print(f"Index type: {type(df.index)}")
    print()
    
    print("FIRST 5 ROWS (head(5)):")
    print("-" * 40)
    # Display first 5 rows and first 10 columns for readability
    sample_df = df.iloc[:5, :10]
    print(sample_df)
    print()
    
    print("SAMPLE VALUES BY SENSOR (first 5 sensors, first 5 timestamps):")
    print("-" * 60)
    for i, col in enumerate(df.columns[:5]):
        values = df[col].head(5)
        print(f"Sensor {col}:")
        for timestamp, value in values.items():
            status = "✓" if not pd.isna(value) else "✗ (NaN)"
            print(f"  {timestamp}: {value} {status}")
        print()
    
    # Show some statistics
    print("VALUE DISTRIBUTION SAMPLES:")
    print("-" * 30)
    sample_values = df.iloc[:100, :100].values.flatten()  # First 100x100 for speed
    sample_values = sample_values[~pd.isna(sample_values)]
    if len(sample_values) > 0:
        print(f"Sample size: {len(sample_values)} values")
        print(f"Min: {sample_values.min()}")
        print(f"Max: {sample_values.max()}")
        print(f"Mean: {sample_values.mean():.2f}")
        print(f"Sample values: {sample_values[:20]}")  # First 20 actual values
    
    return df

def display_metadata_samples():
    """Display head(5) samples from metadata"""
    print("\n" + "="*60)
    print("METADATA - ACTUAL VALUE SAMPLES")
    print("="*60)
    
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    if not os.path.exists(PATH_CONF.metadata_path):
        print(f"❌ Metadata file not found: {PATH_CONF.metadata_path}")
        return None
    
    # Load metadata
    metadata = Metadata.import_from_hdf(PATH_CONF.metadata_path)
    df = metadata.data
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    print("FIRST 5 ROWS (head(5)):")
    print("-" * 40)
    print(df.head(5))
    print()
    
    return df

def display_distances_samples():
    """Display head(5) samples from distance data"""
    print("\n" + "="*60)
    print("DISTANCE DATA - ACTUAL VALUE SAMPLES")
    print("="*60)
    
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    if not os.path.exists(PATH_CONF.distances_path):
        print(f"❌ Distance data file not found: {PATH_CONF.distances_path}")
        return None
    
    # Load distance data
    distances = DistancesImc.import_from_csv(PATH_CONF.distances_path)
    df = distances.data
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    print("FIRST 5 ROWS (head(5)):")
    print("-" * 40)
    print(df.head(5))
    print()
    
    return df

def display_sensor_locations_samples():
    """Display head(5) samples from sensor locations"""
    print("\n" + "="*60)
    print("SENSOR LOCATIONS - ACTUAL VALUE SAMPLES")
    print("="*60)
    
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    if not os.path.exists(PATH_CONF.sensor_locations_path):
        print(f"❌ Sensor locations file not found: {PATH_CONF.sensor_locations_path}")
        return None
    
    # Load sensor locations
    sensor_locations = SensorLocations.import_from_csv(PATH_CONF.sensor_locations_path)
    df = sensor_locations.data
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    print("FIRST 5 ROWS (head(5)):")
    print("-" * 40)
    print(df.head(5))
    print()
    
    return df

def test_dataset_samples():
    """Main test function to display all dataset samples"""
    print("SONGDO TRAFFIC DATASET - VALUE SAMPLES")
    print("="*80)
    print("Displaying head(5) samples from all generated datasets")
    print("="*80)
    
    try:
        # Display samples from each dataset component
        traffic_df = display_traffic_data_samples()
        metadata_df = display_metadata_samples()
        distances_df = display_distances_samples()
        sensor_locations_df = display_sensor_locations_samples()
        
        # Summary
        print("\n" + "="*60)
        print("SAMPLE DISPLAY SUMMARY")
        print("="*60)
        print(f"✓ Traffic Data: {'Loaded' if traffic_df is not None else 'Failed'}")
        print(f"✓ Metadata: {'Loaded' if metadata_df is not None else 'Failed'}")
        print(f"✓ Distance Data: {'Loaded' if distances_df is not None else 'Failed'}")
        print(f"✓ Sensor Locations: {'Loaded' if sensor_locations_df is not None else 'Failed'}")
        
    except Exception as e:
        print(f"\n❌ Error during sample display: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up environment
    os.environ["PYTHONPATH"] = "./src"
    test_dataset_samples()