#!/usr/bin/env python3
"""
Test script to display actual value samples from Nodelink and IMCRTS data
"""
import os
import pandas as pd
import geopandas as gpd
import pickle
from metr.utils import PathConfig

def display_nodelink_samples():
    """Display head(5) samples from Nodelink data"""
    print("="*60)
    print("NODELINK DATA - ACTUAL VALUE SAMPLES")
    print("="*60)
    
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    # Load node data
    print("1. NODE DATA:")
    print("-" * 30)
    if os.path.exists(PATH_CONF.nodelink_node_path):
        try:
            node_data = gpd.read_file(PATH_CONF.nodelink_node_path, encoding='utf-8')
            print(f"Node Data Shape: {node_data.shape}")
            print(f"Node Columns: {list(node_data.columns)}")
            print()
            print("NODE DATA - FIRST 5 ROWS (head(5)):")
            print("-" * 40)
            # Display without geometry column for readability
            display_cols = [col for col in node_data.columns if col != 'geometry']
            sample_data = node_data[display_cols].head(5)
            for idx, row in sample_data.iterrows():
                print(f"Row {idx}:")
                for col, val in row.items():
                    print(f"  {col}: {val}")
                print()
            print()
            
            # Show some sample coordinate info if geometry exists
            if 'geometry' in node_data.columns:
                print("SAMPLE GEOMETRY INFO:")
                for i in range(min(3, len(node_data))):
                    geom = node_data.iloc[i]['geometry']
                    if hasattr(geom, 'x') and hasattr(geom, 'y'):
                        print(f"  Node {i+1}: Point({geom.x:.6f}, {geom.y:.6f})")
                print()
                
        except Exception as e:
            print(f"❌ Error loading node data: {e}")
            node_data = None
    else:
        print(f"❌ Node data file not found: {PATH_CONF.nodelink_node_path}")
        node_data = None
    
    # Load link data
    print("2. LINK DATA:")
    print("-" * 30)
    if os.path.exists(PATH_CONF.nodelink_link_path):
        try:
            link_data = gpd.read_file(PATH_CONF.nodelink_link_path, encoding='utf-8')
            print(f"Link Data Shape: {link_data.shape}")
            print(f"Link Columns: {list(link_data.columns)}")
            print()
            print("LINK DATA - FIRST 5 ROWS (head(5)):")
            print("-" * 40)
            # Display without geometry column for readability
            display_cols = [col for col in link_data.columns if col != 'geometry']
            sample_data = link_data[display_cols].head(5)
            for idx, row in sample_data.iterrows():
                print(f"Row {idx}:")
                for col, val in row.items():
                    print(f"  {col}: {val}")
                print()
            print()
            
            # Show some basic statistics
            if 'LINK_ID' in link_data.columns:
                print(f"Total unique links: {link_data['LINK_ID'].nunique()}")
            if 'LANES' in link_data.columns:
                print(f"Lane count range: {link_data['LANES'].min()}-{link_data['LANES'].max()}")
            if 'MAX_SPD' in link_data.columns:
                print(f"Speed limit range: {link_data['MAX_SPD'].min()}-{link_data['MAX_SPD'].max()} km/h")
            print()
            
        except Exception as e:
            print(f"❌ Error loading link data: {e}")
            link_data = None
    else:
        print(f"❌ Link data file not found: {PATH_CONF.nodelink_link_path}")
        link_data = None
    
    # Load turn data
    print("3. TURN DATA:")
    print("-" * 30)
    if os.path.exists(PATH_CONF.nodelink_turn_path):
        try:
            turn_data = gpd.read_file(PATH_CONF.nodelink_turn_path, encoding='utf-8')
            print(f"Turn Data Shape: {turn_data.shape}")
            print(f"Turn Columns: {list(turn_data.columns)}")
            print()
            print("TURN DATA - FIRST 5 ROWS (head(5)):")
            print("-" * 40)
            sample_data = turn_data.head(5)
            for idx, row in sample_data.iterrows():
                print(f"Row {idx}:")
                for col, val in row.items():
                    print(f"  {col}: {val}")
                print()
            print()
        except Exception as e:
            print(f"❌ Error loading turn data: {e}")
            turn_data = None
    else:
        print(f"❌ Turn data file not found: {PATH_CONF.nodelink_turn_path}")
        turn_data = None
    
    return node_data, link_data, turn_data

def display_imcrts_samples():
    """Display head(5) samples from IMCRTS data"""
    print("\n" + "="*60)
    print("IMCRTS DATA - ACTUAL VALUE SAMPLES")
    print("="*60)
    
    PATH_CONF = PathConfig.from_yaml("config.yaml")
    
    if not os.path.exists(PATH_CONF.imcrts_path):
        print(f"❌ IMCRTS data file not found: {PATH_CONF.imcrts_path}")
        return None
    
    try:
        # Load IMCRTS data from pickle
        print("Loading IMCRTS data from pickle file...")
        with open(PATH_CONF.imcrts_path, 'rb') as f:
            imcrts_data = pickle.load(f)
        
        # Convert to DataFrame if it's not already
        if isinstance(imcrts_data, pd.DataFrame):
            df = imcrts_data
        elif isinstance(imcrts_data, list):
            df = pd.DataFrame(imcrts_data)
        else:
            print(f"Data type: {type(imcrts_data)}")
            df = pd.DataFrame(imcrts_data)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print()
        
        print("FIRST 5 ROWS (head(5)):")
        print("-" * 40)
        sample_data = df.head(5)
        for idx, row in sample_data.iterrows():
            print(f"Row {idx}:")
            for col, val in row.items():
                print(f"  {col}: {val}")
            print()
        print()
        
        # Show some sample data details
        print("DATA SUMMARY:")
        print("-" * 20)
        if 'statDate' in df.columns:
            print(f"Date range: {df['statDate'].min()} to {df['statDate'].max()}")
            print(f"Unique dates: {df['statDate'].nunique()}")
        
        if 'linkId' in df.columns:
            print(f"Unique link IDs: {df['linkId'].nunique()}")
            print(f"Sample link IDs: {list(df['linkId'].unique()[:10])}")
        
        if 'tpcdFlag' in df.columns:
            print(f"Traffic count data points: {len(df)}")
            print(f"Sample traffic counts: {list(df['tpcdFlag'].head(10))}")
        
        # Show detailed sample values
        print("\nDETAILED SAMPLE DATA VALUES:")
        print("-" * 30)
        for i, (idx, row) in enumerate(df.head(3).iterrows()):
            print(f"Record {i+1} (Index: {idx}):")
            for col, val in row.items():
                print(f"  {col}: {val}")
            print()
        
        # Data quality check
        print("DATA QUALITY CHECK:")
        print("-" * 20)
        print("Missing values per column:")
        missing_info = df.isnull().sum()
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} ({missing_count/len(df)*100:.2f}%)")
        
        if missing_info.sum() == 0:
            print("  No missing values found!")
        
    except Exception as e:
        print(f"❌ Error loading IMCRTS data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return df

def test_nodelink_imcrts_samples():
    """Main test function to display Nodelink and IMCRTS dataset samples"""
    print("SONGDO TRAFFIC - NODELINK & IMCRTS DATA SAMPLES")
    print("="*80)
    print("Displaying head(5) samples from Nodelink and IMCRTS datasets")
    print("="*80)
    
    try:
        # Display samples from Nodelink data
        node_data, link_data, turn_data = display_nodelink_samples()
        
        # Display samples from IMCRTS data
        imcrts_df = display_imcrts_samples()
        
        # Summary
        print("\n" + "="*60)
        print("SAMPLE DISPLAY SUMMARY")
        print("="*60)
        print(f"✓ Node Data: {'Loaded' if node_data is not None else 'Failed'}")
        print(f"✓ Link Data: {'Loaded' if link_data is not None else 'Failed'}")
        print(f"✓ Turn Data: {'Loaded' if turn_data is not None else 'Failed'}")
        print(f"✓ IMCRTS Data: {'Loaded' if imcrts_df is not None else 'Failed'}")
        
        # Cross-reference check
        if link_data is not None and imcrts_df is not None:
            print("\nCROSS-REFERENCE CHECK:")
            print("-" * 25)
            if 'LINK_ID' in link_data.columns and 'linkId' in imcrts_df.columns:
                nodelink_ids = set(link_data['LINK_ID'].astype(str))
                imcrts_ids = set(imcrts_df['linkId'].astype(str))
                common_ids = nodelink_ids.intersection(imcrts_ids)
                print(f"Nodelink unique IDs: {len(nodelink_ids)}")
                print(f"IMCRTS unique IDs: {len(imcrts_ids)}")
                print(f"Common IDs: {len(common_ids)}")
                print(f"ID overlap: {len(common_ids)/len(nodelink_ids)*100:.2f}% of Nodelink IDs")
                
                if len(common_ids) > 0:
                    print(f"Sample common IDs: {list(common_ids)[:5]}")
        
    except Exception as e:
        print(f"\n❌ Error during sample display: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up environment
    os.environ["PYTHONPATH"] = "./src"
    test_nodelink_imcrts_samples()