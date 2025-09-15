#!/usr/bin/env python3
"""
Train unified 3-class model: hand, arm, not_hand
Builds on existing hand_cls data with arm class added
"""

import subprocess
from pathlib import Path
from datetime import datetime

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

    print(f"\n📝 Starting training...")
    print("=" * 50)

    # Run training
    try:
        process = subprocess.run(cmd)

        if process.returncode == 0:
            print("\n✅ Training completed!")

            # Find best model
            runs_dir = Path("runs/classify") / run_name
            best_model = runs_dir / "weights/best.pt"

            if best_model.exists():
                # Version management
                existing = list(models_dir.glob("unified_v*.pt"))
                next_version = len(existing) + 1

                # Save versioned
                dest = models_dir / f"unified_v{next_version}.pt"
                import shutil
                shutil.copy(best_model, dest)
                print(f"\n📦 Model saved: {dest}")

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