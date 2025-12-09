"""YOLO model training utilities."""
import os
import yaml
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from ultralytics import YOLO
import cv2


class YOLOTrainer:
    """YOLO model trainer."""

    def __init__(self, camera_id: int, camera_name: str, base_dir: str = "."):
        """
        Initialize YOLO trainer.

        Args:
            camera_id: Camera ID
            camera_name: Camera name
            base_dir: Base directory of the application
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.base_dir = Path(base_dir)
        self.camera_dir = self.base_dir / "cameras" / camera_name
        self.labeled_dir = self.camera_dir / "labeled"
        self.models_dir = self.base_dir / "models"
        self.temp_dir = self.base_dir / "temp" / camera_name

    def prepare_dataset(self) -> str:
        """
        Prepare dataset for training by converting LabelMe format to YOLO format.

        Returns:
            Path to dataset.yaml file
        """
        # Create temp directory structure
        dataset_dir = self.temp_dir / "dataset"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        # Clean and create directories
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Convert annotations
        json_files = list(self.labeled_dir.glob("*.json"))
        image_count = 0

        for json_file in json_files:
            try:
                # Load annotation
                with open(json_file, 'r') as f:
                    annotation = json.load(f)

                image_name = annotation.get('imagePath', json_file.stem + '.jpg')
                image_path = self.labeled_dir / image_name

                if not image_path.exists():
                    continue

                # Copy image
                shutil.copy(image_path, images_dir / image_name)

                # Convert to YOLO format
                yolo_labels = self._convert_to_yolo_format(
                    annotation,
                    image_path
                )

                # Write YOLO label file
                label_file = labels_dir / f"{Path(image_name).stem}.txt"
                with open(label_file, 'w') as f:
                    for label in yolo_labels:
                        f.write(label + '\n')

                image_count += 1

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

        if image_count == 0:
            raise ValueError("No valid training images found")

        # Create dataset.yaml
        dataset_yaml = self._create_dataset_yaml(dataset_dir)

        return str(dataset_yaml)

    def _convert_to_yolo_format(
        self,
        annotation: Dict,
        image_path: Path
    ) -> List[str]:
        """
        Convert LabelMe annotation to YOLO format.

        Args:
            annotation: LabelMe annotation dict
            image_path: Path to image file

        Returns:
            List of YOLO format strings
        """
        # Get image dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img_height, img_width = img.shape[:2]

        yolo_labels = []

        for shape in annotation.get('shapes', []):
            if shape['label'] != 'person':
                continue

            points = shape['points']
            if len(points) != 2:
                continue

            # Get bbox coordinates
            x1, y1 = points[0]
            x2, y2 = points[1]

            # Normalize to 0-1
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height

            # YOLO format: class x_center y_center width height
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return yolo_labels

    def _create_dataset_yaml(self, dataset_dir: Path) -> Path:
        """
        Create dataset.yaml configuration file.

        Args:
            dataset_dir: Dataset directory path

        Returns:
            Path to dataset.yaml
        """
        yaml_content = {
            'path': str(dataset_dir.absolute()),
            'train': 'images',
            'val': 'images',  # Use same for validation (small dataset)
            'nc': 1,
            'names': ['person']
        }

        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)

        return yaml_path

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = "auto"
    ) -> Dict:
        """
        Train YOLO model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            image_size: Input image size
            device: Device to train on

        Returns:
            Training results dict
        """
        # Prepare dataset
        dataset_yaml = self.prepare_dataset()

        # Initialize model
        model = YOLO('yolov8n.pt')

        # Determine device
        if device == "auto":
            device = 0 if os.system("nvidia-smi") == 0 else "cpu"

        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Train
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            project=str(self.models_dir),
            name=f"camera{self.camera_id}",
            exist_ok=True,
            verbose=True,
            patience=10,
            save=True,
            plots=True,
            workers=2  # Reduce workers to avoid shared memory issues
        )

        # Copy best weights to standard location
        best_weights_src = self.models_dir / f"camera{self.camera_id}" / "weights" / "best.pt"
        best_weights_dst = self.models_dir / f"camera{self.camera_id}_weights.pt"

        if best_weights_src.exists():
            shutil.copy(best_weights_src, best_weights_dst)

        # Extract metrics
        metrics = {
            'weights_path': str(best_weights_dst),
            'final_map': float(results.results_dict.get('metrics/mAP50(B)', 0)),
            'final_precision': float(results.results_dict.get('metrics/precision(B)', 0)),
            'final_recall': float(results.results_dict.get('metrics/recall(B)', 0))
        }

        return metrics

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def convert_labelme_to_yolo(
    labelme_json_path: str,
    output_txt_path: str,
    image_path: str
) -> bool:
    """
    Convert single LabelMe JSON to YOLO txt format.

    Args:
        labelme_json_path: Path to LabelMe JSON file
        output_txt_path: Path to output YOLO txt file
        image_path: Path to corresponding image

    Returns:
        True if successful
    """
    try:
        with open(labelme_json_path, 'r') as f:
            annotation = json.load(f)

        img = cv2.imread(image_path)
        if img is None:
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

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True

    except Exception as e:
        print(f"Error converting {labelme_json_path}: {e}")
        return False
