#!/usr/bin/env python3
"""
Additional temporal analysis of the dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from metr.components import TrafficData
from metr.utils import PathConfig

def temporal_analysis():
    """Analyze temporal patterns in the data"""
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    print("\n" + "="*60)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("="*60)
    
    # Load traffic data
    traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
    df = traffic_data.data
    
    # Daily patterns
    df_resampled = df.resample('D').mean()
    daily_means = df_resampled.mean(axis=1)
    
    print(f"Daily Analysis:")
    print(f"  Data available days: {len(daily_means)}")
    print(f"  Average daily traffic: {daily_means.mean():.2f}")
    print(f"  Peak daily traffic: {daily_means.max():.2f} on {daily_means.idxmax().date()}")
    print(f"  Lowest daily traffic: {daily_means.min():.2f} on {daily_means.idxmin().date()}")
    
    # Monthly patterns
    monthly_means = df.groupby(df.index.to_period('M')).mean().mean(axis=1)
    print(f"\nMonthly Analysis:")
    print(f"  Highest traffic month: {monthly_means.idxmax()} ({monthly_means.max():.2f})")
    print(f"  Lowest traffic month: {monthly_means.idxmin()} ({monthly_means.min():.2f})")
    
    # Hourly patterns
    hourly_means = df.groupby(df.index.hour).mean().mean(axis=1)
    print(f"\nHourly Patterns:")
    print(f"  Peak hour: {hourly_means.idxmax()}:00 ({hourly_means.max():.2f})")
    print(f"  Lowest hour: {hourly_means.idxmin()}:00 ({hourly_means.min():.2f})")
    
    # Day of week patterns
    dow_means = df.groupby(df.index.dayofweek).mean().mean(axis=1)
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"\nDay of Week Patterns:")
    for i, day in enumerate(days):
        print(f"  {day}: {dow_means[i]:.2f}")
    
    return df, daily_means, monthly_means, hourly_means, dow_means

def data_quality_analysis():
    """Analyze data quality issues"""
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Load traffic data
    traffic_data = TrafficData.import_from_hdf(PATH_CONF.metr_imc_path)
    df = traffic_data.data
    
    # Zero values analysis
    zero_values = (df == 0).sum().sum()
    total_non_nan = df.notna().sum().sum()
    zero_percentage = (zero_values / total_non_nan) * 100
    
    print(f"Zero Values Analysis:")
    print(f"  Total zero values: {zero_values:,}")
    print(f"  Zero percentage (of non-NaN): {zero_percentage:.2f}%")
    
    # Extreme values analysis
    # Values above 99.9th percentile might be outliers
    all_values = df.values.flatten()
    all_values = all_values[~pd.isna(all_values)]
    percentile_99_9 = np.percentile(all_values, 99.9)
    extreme_values = (all_values > percentile_99_9).sum()
    extreme_percentage = (extreme_values / len(all_values)) * 100
    
    print(f"\nExtreme Values Analysis (>99.9th percentile):")
    print(f"  99.9th percentile threshold: {percentile_99_9:.2f}")
    print(f"  Extreme values count: {extreme_values:,}")
    print(f"  Extreme values percentage: {extreme_percentage:.3f}%")
    
    # Consecutive missing data blocks
    print(f"\nMissing Data Patterns:")
    missing_blocks = []
    for col in df.columns[:10]:  # Check first 10 sensors for efficiency
        is_na = df[col].isna()
        blocks = []
        block_start = None
        for i, is_missing in enumerate(is_na):
            if is_missing and block_start is None:
                block_start = i
            elif not is_missing and block_start is not None:
                blocks.append(i - block_start)
                block_start = None
        if block_start is not None:  # Block continues to end
            blocks.append(len(is_na) - block_start)
        if blocks:
            missing_blocks.extend(blocks)
    
    if missing_blocks:
        print(f"  Average missing block length: {np.mean(missing_blocks):.2f} hours")
        print(f"  Longest missing block: {max(missing_blocks)} hours")
        print(f"  Total missing blocks (sample): {len(missing_blocks)}")

def create_summary_report():
    """Create a comprehensive summary report"""
    print("\n" + "="*80)
    print("GENERATE_RAW_DATASET FUNCTION SUMMARY REPORT")
    print("="*80)
    
    print("\n🎯 PURPOSE:")
    print("   The generate_raw_dataset() function creates a comprehensive traffic dataset")
    print("   for Songdo (Incheon) area by integrating multiple data sources.")
    
    print("\n📊 GENERATED FILES:")
    print("   1. Traffic Data (metr-imc.h5): 345MB - Main time series data")
    print("   2. Metadata (metadata.h5): 1.1MB - Sensor characteristics")
    print("   3. Sensor IDs (metr_ids.txt): 0.02MB - List of sensor identifiers")
    print("   4. Sensor Locations (graph_sensor_locations.csv): 0.11MB - GPS coordinates")
    print("   5. Distance Data (distances_imc.csv): 75.3MB - Inter-sensor distances")
    print("   6. Adjacency Matrix (adj_mx.pkl): 18.3MB - Graph connectivity")
    print("   7. Geographic Data: Node/Link/Turn shapefiles for GIS")
    print("   8. Raw IMCRTS Data (imcrts_data.pkl): 309.6MB - Original API data")
    
    print("\n🔢 KEY STATISTICS:")
    print("   • Sensors: 2,187 traffic sensors")
    print("   • Time Range: 2022-11-01 to 2025-03-10 (29 months)")
    print("   • Data Points: 45,192,168 total measurements")
    print("   • Update Frequency: Hourly")
    print("   • Missing Data: 22.96% (varies by sensor)")
    print("   • Geographic Coverage: All Incheon regions (codes 161-169)")
    
    print("\n🛣️ INFRASTRUCTURE CHARACTERISTICS:")
    print("   • Speed Limits: 30-100 km/h (most common: 50 km/h)")
    print("   • Lane Configuration: 1-5 lanes (most common: 3 lanes)")
    print("   • Network Connections: 1,969,649 distance relationships")
    print("   • Max Sensor Distance: 9km (network-based)")
    print("   • Graph Sparsity: 92.91% (sparse network structure)")
    
    print("\n⚠️ DATA QUALITY NOTES:")
    print("   • No sensors have 0% missing data")
    print("   • 219 sensors have <10% missing data (high quality)")
    print("   • 201 sensors have 50%+ missing data (may need special handling)")
    print("   • Traffic values range from 0 to 995,518 (potential outliers exist)")
    print("   • Median traffic: 62 vehicles/hour, Mean: 197 vehicles/hour")
    
    print("\n🚀 APPLICATIONS:")
    print("   ✓ Traffic flow prediction and forecasting")
    print("   ✓ Anomaly detection and outlier analysis")
    print("   ✓ Spatial-temporal pattern analysis")
    print("   ✓ Graph neural network training")
    print("   ✓ Urban mobility research")
    print("   ✓ Missing data imputation studies")
    
    print("\n📈 RESEARCH VALUE:")
    print("   • Large-scale real-world traffic dataset")
    print("   • Multi-modal data (time series + spatial + metadata)")
    print("   • Suitable for deep learning and ML experiments")
    print("   • Includes natural missing data patterns for robustness testing")
    print("   • Geographic context enables spatial analysis")
    
    print("\n" + "="*80)

def test_detailed_analysis():
    temporal_analysis()
    data_quality_analysis()
    create_summary_report()
