#!/usr/bin/env python3
"""
Three-class training: hand vs arm vs not_hand
Better distinction between hands and arms
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

def train_three_class_detector():
    """Train YOLO classifier for hand/arm/not_hand detection"""

    print("\n🚀 Starting Three-Class Detection Training")
    print("=" * 50)

    # Check data exists
    data_path = Path("data/three_class")
    if not data_path.exists():
        print("❌ Data not found at data/three_class")
        print("   Run first: python3 capture_three_class.py")
        return None

    # Count images
    stats = {}
    for split in ['train', 'val']:
        stats[split] = {}
        for category in ['hand', 'arm', 'not_hand']:
            path = data_path / split / category
            if path.exists():
                stats[split][category] = len(list(path.glob("*.jpg")))
            else:
                stats[split][category] = 0

    print(f"\n📊 Dataset Statistics:")
    print(f"   Training:")
    print(f"      Hands:     {stats['train']['hand']:4d} images")
    print(f"      Arms:      {stats['train']['arm']:4d} images")
    print(f"      Not hands: {stats['train']['not_hand']:4d} images")
    print(f"   Validation:")
    print(f"      Hands:     {stats['val']['hand']:4d} images")
    print(f"      Arms:      {stats['val']['arm']:4d} images")
    print(f"      Not hands: {stats['val']['not_hand']:4d} images")

    total = sum(stats['train'].values()) + sum(stats['val'].values())
    print(f"   Total:        {total:4d} images")

    if total < 100:
        print("\n⚠️  Warning: Very few images. Consider capturing more data.")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Training configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"three_class_{timestamp}"

    # Model configuration
    model_size = 'n'  # nano model for quick training
    epochs = 30  # More epochs for 3 classes
    batch = 32
    patience = 10  # Early stopping patience

    print(f"\n⚙️  Configuration:")
    print(f"   Model: YOLOv8{model_size}-cls (classification)")
    print(f"   Classes: 3 (hand, arm, not_hand)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Early stopping: {patience} epochs")
    print(f"   Data path: {data_path}")

    # Build command
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8{model_size}-cls.pt",
        f"data={data_path}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"name={run_name}",
        f"patience={patience}",
        "save=True",
        "exist_ok=True",
        "plots=True",
        "device=mps" if has_mps() else "device=cpu"
    ]

    print(f"\n📝 Command: {' '.join(cmd)}")
    print("\n" + "=" * 50)
    print("Training started... (this may take several minutes)")
    print("=" * 50 + "\n")

    # Run training
    try:
        process = subprocess.run(cmd)

        if process.returncode == 0:
            print("\n✅ Training completed successfully!")

            # Find the best model
            runs_dir = Path("runs/classify") / run_name
            best_model = runs_dir / "weights/best.pt"

            if best_model.exists():
                # Find next version number for three-class models
                existing_models = list(models_dir.glob("three_class_v*.pt"))
                if existing_models:
                    versions = []
                    for model in existing_models:
                        try:
                            version = int(model.stem.split('_v')[1])
                            versions.append(version)
                        except:
                            pass
                    next_version = max(versions) + 1 if versions else 1
                else:
                    next_version = 1

                # Copy to models directory with version
                dest = models_dir / f"three_class_v{next_version}.pt"
                import shutil
                shutil.copy(best_model, dest)
                print(f"\n📦 Model saved to: {dest}")

                # Also create a "latest" three-class model
                latest = models_dir / "three_class_detector.pt"
                shutil.copy(best_model, latest)
                print(f"   Also updated: {latest} (latest)")
                print(f"   Training plots: {runs_dir}")

                # Show performance metrics if available
                if (runs_dir / "results.csv").exists():
                    print("\n📈 Training Metrics:")
                    # Read last line of results.csv for final metrics
                    with open(runs_dir / "results.csv", 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            headers = lines[0].strip().split(',')
                            values = lines[-1].strip().split(',')
                            for h, v in zip(headers[:5], values[:5]):
                                print(f"   {h:15s}: {float(v):6.4f}")

                print("\n🎥 To test the THREE-CLASS detection:")
                print("   python3 live_demo_three_class.py")
                print("\n   Color coding:")
                print("   - 🔴 RED = HAND detected")
                print("   - 🟡 YELLOW = ARM detected")
                print("   - ⚫ GRAY = Neither")

                return dest
            else:
                print("⚠️  Could not find best.pt model file")
        else:
            print(f"\n❌ Training failed with code {process.returncode}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training error: {e}")

    return None

def has_mps():
    """Check if Apple Silicon MPS is available"""
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False

def test_model(model_path):
    """Quick test of the trained model"""
    if not model_path or not model_path.exists():
        print("❌ No model to test")
        return

    print(f"\n🧪 Testing model: {model_path}")

    # Test with sample images from each class
    data_path = Path("data/three_class/val")

    for category in ['hand', 'arm', 'not_hand']:
        cat_path = data_path / category
        if cat_path.exists():
            sample_images = list(cat_path.glob("*.jpg"))[:2]

            if sample_images:
                print(f"\n   Testing {category} samples...")

                for img in sample_images:
                    cmd = [
                        "yolo", "classify", "predict",
                        f"model={model_path}",
                        f"source={img}",
                        "save=False",
                        "verbose=False"
                    ]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        # Parse output for prediction
                        if "hand" in result.stdout.lower():
                            if category == 'hand':
                                print(f"      ✅ Correctly identified as hand")
                            else:
                                print(f"      ❌ Misidentified as hand (actual: {category})")
                        elif "arm" in result.stdout.lower():
                            if category == 'arm':
                                print(f"      ✅ Correctly identified as arm")
                            else:
                                print(f"      ❌ Misidentified as arm (actual: {category})")
                        elif "not_hand" in result.stdout.lower():
                            if category == 'not_hand':
                                print(f"      ✅ Correctly identified as not_hand")
                            else:
                                print(f"      ❌ Misidentified as not_hand (actual: {category})")
                    except:
                        print(f"      ⚠️  Test failed for {img.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train three-class detector")
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model only')
    args = parser.parse_args()

    if args.test_only:
        model = Path("models/three_class_detector.pt")
        if model.exists():
            test_model(model)
        else:
            print("No model found. Train first!")
    else:
        model_path = train_three_class_detector()
        if model_path:
            test_model(model_path)
            print("\n🎉 Three-class detector ready!")
            print("\nNext steps:")
            print("1. Test live: python3 live_demo_three_class.py")
            print("2. Collect more data: python3 capture_three_class.py")
            print("3. Deploy: python3 deploy_huggingface.py")