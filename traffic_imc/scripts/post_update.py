#!/usr/bin/env python3
"""
Post-update script to copy STGCN models file
Copies ../third_party/stgcn/model/models.py to ./src/metr_val/models/stgcn.py
"""
import shutil
import os
from pathlib import Path

def copy_stgcn_models():
    """Copy STGCN models file to destination with new name"""
    # Get the project root (where pyproject.toml is located)
    project_root = Path(__file__).parent.parent
    stgcn_model_file_relative_path = "../third_party/stgcn/model/models.py"
    stgcn_model_target_relative_dir_path = "src/metr_val/models"
    stgcn_model_target_relative_path = f"{stgcn_model_target_relative_dir_path}/stgcn.py"
    
    # Source file path (relative to project root)
    source_file = project_root / stgcn_model_file_relative_path

    # Destination directory and file
    dest_dir = project_root / stgcn_model_target_relative_dir_path
    dest_file = project_root / stgcn_model_target_relative_path
    
    try:
        # Resolve absolute paths
        source_file = source_file.resolve()
        dest_file = dest_file.resolve()
        
        # Check if source file exists
        if not source_file.exists():
            print(f"❌ Source file not found: {source_file}")
            return False
        
        # Create destination directory if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_file, dest_file)
        
        print(f"✅ STGCN model successfully copied:")
        print(f"   From: {stgcn_model_file_relative_path}")
        print(f"   To:   ./{stgcn_model_target_relative_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying STGCN models file: {e}")
        return False

def main():
    """Main function for post-update operations"""
    print("=== PDM Post-Update Script ===")
    
    # Copy STGCN models file
    if not copy_stgcn_models():
        print("❌ STGCN models file copy failed")
    
    print("=== Post-Update Complete ===")

if __name__ == "__main__":
    main()
