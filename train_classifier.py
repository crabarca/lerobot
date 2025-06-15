import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import argparse

def setup_classification_dataset():
    # Get the absolute path of the current directory
    current_dir = Path("C:/Users/Zakariea/Documents/GitHub/lerobot")
    print(f"Current working directory: {current_dir}")
    
    # Create dataset directory structure
    dataset_dir = current_dir / "classification_dataset"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    # Remove existing dataset directory if it exists
    if dataset_dir.exists():
        print(f"Removing existing dataset directory: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    # Create directories
    for dir_path in [train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "paper_ball").mkdir(exist_ok=True)
        (dir_path / "bottle_cap").mkdir(exist_ok=True)
    
    # Define dataset paths
    datasets = {
        "paper_ball": current_dir / "Paperball detection.v1i.yolov8",
        "bottle_cap": current_dir / "korki.v1i.yolov8"
    }
    
    total_images = 0
    for class_name, dataset_path in datasets.items():
        print(f"\nProcessing dataset: {dataset_path}")
        if not dataset_path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            continue
            
        # Get all images from the dataset
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(list(dataset_path.glob(f"**/{ext}")))
        
        if not images:
            print(f"Warning: No images found in {dataset_path}")
            continue
            
        print(f"Found {len(images)} images in {dataset_path}")
        total_images += len(images)
        
        # Split into train/val (80/20)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)
            
        print(f"Copied {len(train_images)} images to train/{class_name}")
        print(f"Copied {len(val_images)} images to val/{class_name}")
    
    if total_images == 0:
        raise RuntimeError("No images found in any dataset!")
    
    print(f"\nTotal dataset structure:")
    print(f"  - Total images: {total_images}")
    print(f"  - Train: {len(list(train_dir.glob('**/*.*')))} images")
    print(f"  - Val: {len(list(val_dir.glob('**/*.*')))} images")
    
    # Verify the dataset structure
    if not (train_dir / "paper_ball").exists() or not (train_dir / "bottle_cap").exists():
        raise RuntimeError("Train directory structure is incorrect!")
    if not (val_dir / "paper_ball").exists() or not (val_dir / "bottle_cap").exists():
        raise RuntimeError("Validation directory structure is incorrect!")
    
    return dataset_dir

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Classifier Trainer")
    parser.add_argument('--rebuild-dataset', action='store_true', help='Rebuild the classification dataset from source folders')
    args = parser.parse_args()

    if args.rebuild_dataset:
        dataset_dir = setup_classification_dataset()
    else:
        dataset_dir = Path("C:/Users/Zakariea/Documents/GitHub/lerobot/classification_dataset")

    train_classifier(dataset_dir)

def train_classifier(dataset_dir):
    try:
        # Load a pretrained YOLOv8 classification model
        model = YOLO('yolov8n-cls.pt')
        # Train the model
        results = model.train(
            data=str(dataset_dir).replace('\\', '/'),  # Directory, not YAML!
            epochs=100,
            imgsz=640,
            batch=16,
            name='so101_classifier',
            patience=20,
            save=True,
            device='0',
            project=str(dataset_dir.parent).replace('\\', '/'),
            exist_ok=True
        )
        # Validate the model
        model.val()
        # Export the model to ONNX format
        model.export(format='onnx')
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
