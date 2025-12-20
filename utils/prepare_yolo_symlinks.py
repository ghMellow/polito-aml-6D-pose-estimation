"""
Prepare YOLO Dataset with Symbolic Links

Creates YOLO-format dataset structure using symbolic links instead of copying files.
This is MUCH faster (~2 seconds vs 5+ minutes) and saves ~4.5GB disk space.
"""

import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.bbox_utils import convert_bbox_to_yolo_format


def prepare_yolo_dataset_symlinks(dataset_root, output_root, use_symlinks=True):
    """
    Prepare YOLO dataset using symbolic links (or hard copy as fallback).
    
    Args:
        dataset_root: Path to Linemod_preprocessed
        output_root: Output directory for YOLO format
        use_symlinks: If True, create symlinks; if False, copy files
    
    Returns:
        dict: Statistics (train/val counts, time taken)
    """
    import time
    from PIL import Image
    import shutil
    
    start_time = time.time()
    output_root = Path(output_root)
    
    # Create directory structure
    for split in ['train', 'val']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(dataset_root) / 'data'
    stats = {'train': 0, 'val': 0, 'skipped': 0, 'clipped': 0}
    
    print(f"🔄 Preparing YOLO dataset with {'symlinks' if use_symlinks else 'file copies'}...")
    print(f"   Input: {dataset_root}")
    print(f"   Output: {output_root}")
    
    # Process each object folder
    folders = [f for f in sorted(data_dir.iterdir()) if f.is_dir()]
    
    for folder in tqdm(folders, desc="Processing folders"):
        folder_id = int(folder.name)
        
        # Skip folders not in mapping (03, 07 missing)
        if folder_id not in Config.FOLDER_ID_TO_CLASS_ID:
            continue
        
        class_id = Config.FOLDER_ID_TO_CLASS_ID[folder_id]
        
        gt_path = folder / 'gt.yml'
        if not gt_path.exists():
            continue
        
        with open(gt_path, 'r') as f:
            gt_data = yaml.safe_load(f)
        
        # Process train and test splits
        for split_name in ['train', 'test']:
            split_file = folder / f'{split_name}.txt'
            if not split_file.exists():
                continue
            
            with open(split_file, 'r') as f:
                sample_ids = [int(line.strip()) for line in f if line.strip()]
            
            # YOLO uses 'val' instead of 'test'
            yolo_split = 'train' if split_name == 'train' else 'val'
            
            for sample_id in sample_ids:
                # Source image path
                img_path = folder / 'rgb' / f'{sample_id:04d}.png'
                if not img_path.exists():
                    stats['skipped'] += 1
                    continue
                
                # Get image dimensions (needed for bbox conversion)
                try:
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    img.close()
                except Exception as e:
                    print(f"⚠️  Error reading {img_path}: {e}")
                    stats['skipped'] += 1
                    continue
                
                # Unique filename
                new_name = f'{folder_id:02d}_{sample_id:04d}'
                
                # Destination paths
                dst_img = output_root / 'images' / yolo_split / f'{new_name}.png'
                label_path = output_root / 'labels' / yolo_split / f'{new_name}.txt'
                
                # Create symlink or copy image
                if dst_img.exists():
                    dst_img.unlink()  # Remove existing symlink/file
                
                if use_symlinks:
                    try:
                        # Create relative symlink for portability
                        os.symlink(img_path.resolve(), dst_img)
                    except OSError:
                        # Fallback to copying if symlinks not supported
                        shutil.copy2(img_path, dst_img)
                else:
                    shutil.copy2(img_path, dst_img)
                
                # Create YOLO label file
                if sample_id not in gt_data:
                    # No objects, create empty file
                    label_path.touch()
                else:
                    obj_list = gt_data[sample_id]
                    if not isinstance(obj_list, list):
                        obj_list = [obj_list]

                    with open(label_path, 'w') as label_file:
                        for obj in obj_list:
                            bbox = obj['obj_bb']  # [x, y, w, h]
                            # Convert to YOLO format
                            x_c, y_c, w_n, h_n = convert_bbox_to_yolo_format(
                                bbox, img_width, img_height
                            )

                            # Validate normalized coordinates — clip to [0,1] and log if necessary
                            orig_vals = (x_c, y_c, w_n, h_n)
                            out_of_bounds = any((v < 0.0 or v > 1.0) for v in orig_vals)
                            if out_of_bounds:
                                stats['clipped'] += 1
                                print(f"⚠️  Out-of-bounds bbox for {new_name}.png: original={orig_vals}")
                                # Clip values to [0,1]
                                x_c = min(max(x_c, 0.0), 1.0)
                                y_c = min(max(y_c, 0.0), 1.0)
                                w_n = min(max(w_n, 0.0), 1.0)
                                h_n = min(max(h_n, 0.0), 1.0)

                            # Write: class_id x_center y_center width height
                            label_file.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                
                stats[yolo_split] += 1
    
    elapsed_time = time.time() - start_time
    stats['time_seconds'] = elapsed_time
    
    return stats


def create_data_yaml(output_root, dataset_root):
    """
    Create data.yaml configuration file for YOLO training.
    """
    yaml_path = output_root / 'data.yaml'

    # Derive class list from Config to keep names and indices aligned with the dataset mapping
    class_items = sorted(
        Config.LINEMOD_OBJECTS.values(), key=lambda obj: obj['yolo_class']
    )
    names_block = "\n".join(
        [f"  {obj['yolo_class']}: {obj['name']}" for obj in class_items]
    )

    yaml_content = f"""# LineMOD Dataset Configuration for YOLO
# Auto-generated by prepare_yolo_symlinks.py

path: {str(output_root.resolve())}
train: images/train
val: images/val
nc: {len(class_items)}

# Classes derived from Config (zero-based for YOLO)
names:
{names_block}
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset with symlinks')
    parser.add_argument('--data_dir', type=str, default=str(Config.DATASETS_DIR),
                       help='Path to Linemod_preprocessed root')
    parser.add_argument('--output_dir', type=str, 
                       default=str(Config.DATASETS_DIR / 'yolo_symlinks'),
                       help='Output directory for YOLO format')
    parser.add_argument('--no-symlinks', action='store_true',
                       help='Copy files instead of creating symlinks')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("YOLO Dataset Preparation with Symbolic Links")
    print("=" * 80)
    
    # Prepare dataset
    stats = prepare_yolo_dataset_symlinks(
        dataset_root=args.data_dir,
        output_root=args.output_dir,
        use_symlinks=not args.no_symlinks
    )
    
    # Create data.yaml
    yaml_path = create_data_yaml(Path(args.output_dir), args.data_dir)
    
    # Print summary
    print(f"\n✅ Dataset prepared successfully!")
    print(f"\n📊 Statistics:")
    print(f"   Train images: {stats['train']}")
    print(f"   Val images: {stats['val']}")
    print(f"   Skipped: {stats['skipped']}")
    print(f"   Time: {stats['time_seconds']:.2f} seconds")
    print(f"   Speed: {(stats['train'] + stats['val']) / stats['time_seconds']:.0f} images/second")
    print(f"\n📄 Configuration file: {yaml_path}")
    print(f"\n💡 Use this data.yaml for training:")
    print(f"   python scripts/train_yolo.py --data_yaml {yaml_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
