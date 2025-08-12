#!/usr/bin/env python3
"""
Post-update script to copy STGCN model folder
Copies ../third_party/stgcn/model folder to ./src/metr_val/models/stgcn
"""
import shutil
import os
from pathlib import Path

def fix_imports_in_models_py(models_py_path):
    """Fix import statements in models.py to use relative imports"""
    try:
        # Read the file content
        with open(models_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace "from model import" with "from . import"
        original_content = content
        content = content.replace("from model import", "from . import")
        
        # Only write back if changes were made
        if content != original_content:
            with open(models_py_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"ℹ️  No import statements to fix in models.py")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing imports in models.py: {e}")
        return False

def copy_stgcn_models():
    """Copy STGCN model files (excluding __init__.py) to destination folder"""
    # Get the project root (where pyproject.toml is located)
    project_root = Path(__file__).parent.parent
    stgcn_model_folder_relative_path = "../third_party/stgcn/model"
    stgcn_model_target_relative_path = "src/metr_val/models/stgcn"
    
    # Source folder path (relative to project root)
    source_folder = project_root / stgcn_model_folder_relative_path
    dest_folder = project_root / stgcn_model_target_relative_path
    
    try:
        # Resolve absolute paths
        source_folder = source_folder.resolve()
        dest_folder = dest_folder.resolve()
        
        # Check if source folder exists
        if not source_folder.exists():
            print(f"❌ Source folder not found: {source_folder}")
            return False
        
        # Check if destination folder exists (it should exist according to requirements)
        if not dest_folder.exists():
            print(f"❌ Destination folder not found: {dest_folder}")
            return False
        
        # Copy files from source folder excluding __init__.py (existing files will be overwritten)
        copied_files = []
        for item in source_folder.iterdir():
            if item.is_file() and item.name != "__init__.py":
                dest_file = dest_folder / item.name
                shutil.copy2(item, dest_file)
                copied_files.append(item.name)
            elif item.is_dir():
                dest_dir = dest_folder / item.name
                shutil.copytree(item, dest_dir)
                copied_files.append(f"{item.name}/")
        
        # Fix import statements in models.py
        models_py_path = dest_folder / "models.py"
        if models_py_path.exists():
            fix_imports_in_models_py(models_py_path)
        
        print(f"✅ STGCN model files successfully copied:")
        print(f"   From: {stgcn_model_folder_relative_path}")
        print(f"   To:   ./{stgcn_model_target_relative_path}")
        print(f"   Copied files: {', '.join(copied_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying STGCN model files: {e}")
        return False

def main():
    """Main function for post-update operations"""
    print("=== PDM Post-Update Script ===")
    
    # Copy STGCN model files
    if not copy_stgcn_models():
        print("❌ STGCN model files copy failed")
    
    print("=== Post-Update Complete ===")

if __name__ == "__main__":
    main()
