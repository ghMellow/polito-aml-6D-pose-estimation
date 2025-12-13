"""
Dataset Download Script for LineMOD Preprocessed Dataset

This script downloads the preprocessed LineMOD dataset from Google Drive
and extracts it to the data/ directory.

Usage:
    python utils/download_dataset.py
"""

import os
import zipfile
import gdown
from pathlib import Path

# Import Config for default paths
try:
    from config import Config
    DEFAULT_OUTPUT_DIR = str(Config.DATASETS_DIR)
except ImportError:
    DEFAULT_OUTPUT_DIR = './data'


def download_linemod_dataset(output_dir=None):
    """
    Download and extract LineMOD preprocessed dataset.
    
    Args:
        output_dir (str): Directory where dataset will be saved (default: from Config.DATASETS_DIR)
    """
    # Use Config default if not specified
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Google Drive file ID for the LineMOD preprocessed dataset
    file_url = 'https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=sharing'
    zip_path = output_path / 'Linemod_preprocessed.zip'
    
    print("📥 Downloading LineMOD preprocessed dataset...")
    print(f"   URL: {file_url}")
    print(f"   Destination: {zip_path}")
    
    try:
        # Download the file using gdown
        gdown.download(file_url, str(zip_path), quiet=False, fuzzy=True)
        print("✅ Download completed!")
        
        # Extract the zip file
        print("\n📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print("✅ Extraction completed!")
        
        # Remove zip file to save space
        print("\n🗑️  Removing zip file...")
        os.remove(zip_path)
        print("✅ Cleanup completed!")
        
        print(f"\n✨ Dataset successfully downloaded and extracted to: {output_path}")
        print("\n📁 Dataset structure:")
        # Show directory structure
        for root, dirs, files in os.walk(output_path):
            level = root.replace(str(output_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            if level < 2:  # Only show first 2 levels
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show max 5 files per directory
                    print(f'{subindent}{file}')
                if len(files) > 5:
                    print(f'{subindent}... and {len(files) - 5} more files')
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download LineMOD dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help=f'Directory where dataset will be saved (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    download_linemod_dataset(args.output_dir)
