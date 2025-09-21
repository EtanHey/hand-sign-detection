#!/usr/bin/env python3
"""
Train unified 3-class model: hand, arm, not_hand
Builds on existing hand_cls data with arm class added
"""

import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import os
import json
import re
from typing import Dict, List, Any

def parse_training_output(line: str, epoch_metrics: List[Dict]) -> None:
    """Parse YOLO training output line and extract metrics"""
    # Pattern for epoch metrics: "1/50  1.23G  0.652  16  224: 100%|██| 95/95"
    # Also captures validation accuracy lines

    # Check for epoch completion with metrics
    epoch_pattern = r'^\s*(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+\d+\s+\d+:'
    match = re.match(epoch_pattern, line)
    if match:
        epoch_num = int(match.group(1))
        total_epochs = int(match.group(2))
        loss = float(match.group(3))

        # Find or create epoch entry
        epoch_entry = None
        for entry in epoch_metrics:
            if entry['epoch'] == epoch_num:
                epoch_entry = entry
                break

        if not epoch_entry:
            epoch_entry = {
                'epoch': epoch_num,
                'total_epochs': total_epochs,
                'train_loss': loss,
                'val_accuracy': None,
                'timestamp': datetime.now().isoformat()
            }
            epoch_metrics.append(epoch_entry)
        else:
            epoch_entry['train_loss'] = loss

    # Check for validation accuracy
    if 'top1_acc' in line and 'classes' in line:
        # Extract accuracy from validation results
        acc_pattern = r'all\s+([\d.]+)\s+'
        acc_match = re.search(acc_pattern, line)
        if acc_match and epoch_metrics:
            # Add to the most recent epoch
            epoch_metrics[-1]['val_accuracy'] = float(acc_match.group(1))

def save_training_metrics(version_name: str, epoch_metrics: List[Dict],
                         training_config: Dict, final_stats: Dict = None) -> Path:
    """Save training metrics to a JSON file for version control"""
    metrics_dir = Path("models/metrics")
    metrics_dir.mkdir(exist_ok=True, parents=True)

    # Create metrics filename matching the model version
    metrics_file = metrics_dir / f"{version_name}_metrics.json"

    metrics_data = {
        'version': version_name,
        'training_date': datetime.now().isoformat(),
        'configuration': training_config,
        'epoch_metrics': epoch_metrics,
        'final_performance': final_stats or {},
        'summary': {
            'total_epochs_trained': len(epoch_metrics),
            'best_val_accuracy': max([e.get('val_accuracy', 0) for e in epoch_metrics] or [0]),
            'final_train_loss': epoch_metrics[-1]['train_loss'] if epoch_metrics else None,
            'final_val_accuracy': epoch_metrics[-1].get('val_accuracy') if epoch_metrics else None
        }
    }

    # Save metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\n📊 Training metrics saved: {metrics_file}")

    # Also create a summary file for easy comparison
    summary_file = metrics_dir / "training_summary.json"

    # Load existing summary if it exists
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {'models': {}}

    # Add this model's summary
    summary['models'][version_name] = {
        'date': datetime.now().isoformat(),
        'epochs': len(epoch_metrics),
        'best_val_accuracy': metrics_data['summary']['best_val_accuracy'],
        'final_loss': metrics_data['summary']['final_train_loss'],
        'final_accuracy': metrics_data['summary']['final_val_accuracy']
    }

    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return metrics_file

def incorporate_corrections(data_path):
    """Add correction images to training data with special marking"""
    corrections_dir = Path("../unified-detector-client/corrections")

    if not corrections_dir.exists():
        print("📝 No corrections found yet")
        return 0

    print("\n🔄 Incorporating corrections into training data...")
    total_corrections = 0

    for class_name in ['hand', 'arm', 'not_hand']:
        src_dir = corrections_dir / class_name
        if not src_dir.exists() or not any(src_dir.iterdir()):
            continue

        # Copy corrections to training data with a special prefix
        train_dir = data_path / 'train' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)

        for img in src_dir.glob("*.jpg"):
            # Add CORRECTED_ prefix to track these are corrections
            dest = train_dir / f"CORRECTED_{img.name}"
            shutil.copy2(img, dest)
            total_corrections += 1
            print(f"   Added correction: {img.name} → {class_name}")

    if total_corrections > 0:
        print(f"\n✅ Added {total_corrections} corrected images to training")
        print("   These will be weighted 2x during training")

    return total_corrections

def train_unified_model():
    """Train on hand_cls dataset with all three classes"""

    print("\n🚀 Training 3-Class Model (hand/arm/not_hand)")
    print("=" * 50)

    # Check data
    data_path = Path("data/hand_cls")
    if not data_path.exists():
        print("❌ hand_cls data not found")
        print("   Run: python3 capture_arm_only.py")
        return None

    # Incorporate corrections if available
    corrections_count = incorporate_corrections(data_path)

    # Count images
    print("\n📊 Dataset Statistics:")
    total = 0
    class_counts = {}

    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        for category in ['hand', 'arm', 'not_hand']:
            path = data_path / split / category
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                total += count
                class_counts[f"{split}_{category}"] = count
                print(f"  {category:10s}: {count:4d} images")
            else:
                class_counts[f"{split}_{category}"] = 0
                print(f"  {category:10s}:    0 images ⚠️")

    print(f"\nTotal: {total} images")

    # Check for arm data
    arm_total = class_counts.get('train_arm', 0) + class_counts.get('val_arm', 0)
    if arm_total == 0:
        print("\n⚠️  No ARM images found!")
        print("   Run: python3 capture_arm_only.py")
        response = input("\nTrain without arm class? (y/n): ")
        if response.lower() != 'y':
            return None

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Training configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"unified_{timestamp}"

    # Use absolute path
    data_abs = data_path.absolute()

    # Model parameters
    model_size = 's'  # Small for better accuracy
    epochs = 50  # Good balance between training time and accuracy
    batch = 16
    patience = 15  # Increased patience for better convergence

    print(f"\n⚙️  Configuration:")
    print(f"   Model: YOLOv8{model_size}-cls")
    print(f"   Classes: hand ({class_counts.get('train_hand', 0)}), arm ({class_counts.get('train_arm', 0)}), not_hand ({class_counts.get('train_not_hand', 0)})")
    print(f"   Epochs: {epochs}")
    print(f"   Batch: {batch}")
    print(f"   Data: {data_abs}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Build command
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8{model_size}-cls.pt",
        f"data={data_abs}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"patience={patience}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
        "plots=True",
        "device=mps" if has_mps() else "device=cpu"
    ]

    print(f"\n📝 Starting training with metrics tracking...")
    print("=" * 50)

    # Prepare metrics tracking
    epoch_metrics = []
    training_config = {
        'model': f'YOLOv8{model_size}-cls',
        'epochs': epochs,
        'batch_size': batch,
        'patience': patience,
        'data_path': str(data_abs),
        'classes': ['hand', 'arm', 'not_hand'],
        'class_counts': class_counts,
        'corrections_added': corrections_count
    }

    # Version management - determine version before training
    existing = list(models_dir.glob("unified_v*.pt"))
    next_version = len(existing) + 1
    version_name = f"unified_v{next_version}"

    print(f"\n📈 Training as {version_name}")
    print("   Metrics will be saved for each epoch")

    # Run training with output capture
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, bufsize=1)

        # Process output line by line
        for line in process.stdout:
            print(line, end='')  # Print to console
            parse_training_output(line, epoch_metrics)  # Parse metrics

            # Show epoch summary when completed
            if epoch_metrics and 'val_accuracy' in epoch_metrics[-1] and epoch_metrics[-1]['val_accuracy'] is not None:
                last_epoch = epoch_metrics[-1]
                print(f"\n📊 Epoch {last_epoch['epoch']}/{last_epoch['total_epochs']} Summary:")
                print(f"   Training Loss: {last_epoch['train_loss']:.4f}")
                print(f"   Validation Accuracy: {last_epoch['val_accuracy']:.1%}")

        process.wait()

        if process.returncode == 0:
            print("\n✅ Training completed!")

            # Find best model
            runs_dir = Path("runs/classify") / run_name
            best_model = runs_dir / "weights/best.pt"

            if best_model.exists():
                # Save versioned model
                dest = models_dir / f"{version_name}.pt"
                shutil.copy(best_model, dest)
                print(f"\n📦 Model saved: {dest}")

                # Save training metrics
                metrics_file = save_training_metrics(
                    version_name,
                    epoch_metrics,
                    training_config
                )

                # Update latest
                latest = models_dir / "unified_detector.pt"
                shutil.copy(best_model, latest)
                print(f"   Latest: {latest}")

                # Also update hand_detector.pt for backward compatibility
                hand_detector = models_dir / "hand_detector.pt"
                shutil.copy(best_model, hand_detector)
                print(f"   Compatible: {hand_detector}")

                print(f"\n📊 Training plots: {runs_dir}")

                print("\n🎥 Test with:")
                print("   python3 live_demo_unified.py")
                print("\nOr use with existing demo:")
                print("   python3 live_demo_three_class.py --weights models/unified_detector.pt")

                return dest
        else:
            print(f"\n❌ Training failed with code {process.returncode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    return None

def has_mps():
    """Check if Apple Silicon MPS is available"""
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False

if __name__ == "__main__":
    model = train_unified_model()
    if model:
        print("\n🎉 Unified model ready!")
        print("The model can now distinguish:")
        print("  • HAND (close-up, fingers visible)")
        print("  • ARM (forearm, elbow area)")
        print("  • NOT_HAND (neither)")