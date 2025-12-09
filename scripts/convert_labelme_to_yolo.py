#!/usr/bin/env python3
"""
Utility script to convert LabelMe JSON annotations to YOLO format.

Usage:
    python convert_labelme_to_yolo.py --input_dir cameras/camera1/labeled --output_dir dataset
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import cv2


def convert_labelme_to_yolo(json_path: Path, output_txt_path: Path, image_path: Path) -> bool:
    """
    Convert single LabelMe JSON to YOLO txt format.

    Args:
        json_path: Path to LabelMe JSON file
        output_txt_path: Path to output YOLO txt file
        image_path: Path to corresponding image

    Returns:
        True if successful
    """
    try:
        with open(json_path, 'r') as f:
            annotation = json.load(f)

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Cannot read image {image_path}")
            return False

        img_height, img_width = img.shape[:2]

        with open(output_txt_path, 'w') as f:
            for shape in annotation.get('shapes', []):
                if shape['label'] != 'person':
                    continue

                points = shape['points']
                if len(points) != 2:
                    continue

                x1, y1 = points[0]
                x2, y2 = points[1]

                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height

                # YOLO format: class x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True

    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON to YOLO format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with LabelMe JSONs and images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for YOLO dataset')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Create output directories
    images_train_dir = output_dir / 'images' / 'train'
    images_val_dir = output_dir / 'images' / 'val'
    labels_train_dir = output_dir / 'labels' / 'train'
    labels_val_dir = output_dir / 'labels' / 'val'

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = list(input_dir.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Split into train/val
    split_idx = int(len(json_files) * args.train_split)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Convert train files
    converted = 0
    for json_file in train_files:
        # Get corresponding image
        image_name = json_file.stem
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            test_path = input_dir / f"{image_name}{ext}"
            if test_path.exists():
                image_path = test_path
                break

        if not image_path:
            print(f"Warning: No image found for {json_file}")
            continue

        # Convert to YOLO format
        output_txt = labels_train_dir / f"{image_name}.txt"
        if convert_labelme_to_yolo(json_file, output_txt, image_path):
            # Copy image
            shutil.copy(image_path, images_train_dir / image_path.name)
            converted += 1

    # Convert val files
    for json_file in val_files:
        image_name = json_file.stem
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            test_path = input_dir / f"{image_name}{ext}"
            if test_path.exists():
                image_path = test_path
                break

        if not image_path:
            continue

        output_txt = labels_val_dir / f"{image_name}.txt"
        if convert_labelme_to_yolo(json_file, output_txt, image_path):
            shutil.copy(image_path, images_val_dir / image_path.name)
            converted += 1

    print(f"\nConversion complete! Converted {converted} images")

    # Create dataset.yaml
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val

nc: 1
names: ['person']
"""

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created {yaml_path}")
    print("\nYou can now train with:")
    print(f"  yolo train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")


if __name__ == '__main__':
    main()
