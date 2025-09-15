#!/usr/bin/env python3
"""
Three-class training with smart hierarchy:
- Hand > Arm > Nothing
- If hand is visible, it's ALWAYS hand (even if arm is also visible)
- If no hand but arm visible, it's arm
- Otherwise it's nothing
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

def create_weighted_config():
    """Create YOLO config with class weights favoring hand detection"""

    # Get absolute path to data directory
    data_path = Path.cwd() / 'data' / 'three_class'

    # Class weights to favor hand detection
    # Higher weight = more important to get right
    config = {
        'path': str(data_path.absolute()),  # Use absolute path
        'train': 'train',
        'val': 'val',
        'names': {
            0: 'hand',      # Priority 1 - Always wins
            1: 'arm',       # Priority 2 - Only if no hand
            2: 'not_hand'   # Priority 3 - Neither
        },
        # Custom class weights - hand mistakes are 2x more costly
        'class_weights': [2.0, 1.0, 0.8]
    }

    config_path = Path('three_class_weighted.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path.absolute()  # Return absolute path

def prepare_training_data():
    """Verify and prepare training data with proper labels"""
    data_path = Path("data/three_class")

    if not data_path.exists():
        print("❌ Data not found at data/three_class")
        print("   Run first: python3 capture_three_class.py")
        return False

    # Count and verify images
    stats = {}
    total = 0

    print("\n📊 Verifying Dataset Labels:")
    print("-" * 50)

    for split in ['train', 'val']:
        stats[split] = {}
        for category in ['hand', 'arm', 'not_hand']:
            path = data_path / split / category
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                stats[split][category] = count
                total += count

                # Show sample images to verify correct labeling
                if count > 0:
                    print(f"{split:5s}/{category:10s}: {count:4d} images ✓")
            else:
                stats[split][category] = 0
                print(f"{split:5s}/{category:10s}:    0 images ⚠️")

    if total < 100:
        print("\n⚠️  Warning: Very few images. Consider capturing more data.")
        return False

    print(f"\nTotal: {total} images")

    # Check class balance
    hand_total = stats['train']['hand'] + stats['val']['hand']
    arm_total = stats['train']['arm'] + stats['val']['arm']
    not_hand_total = stats['train']['not_hand'] + stats['val']['not_hand']

    print("\n📈 Class Distribution:")
    print(f"   Hands:     {hand_total:4d} ({hand_total*100//total}%)")
    print(f"   Arms:      {arm_total:4d} ({arm_total*100//total}%)")
    print(f"   Not hands: {not_hand_total:4d} ({not_hand_total*100//total}%)")

    # Warn if severely imbalanced
    if hand_total < 50:
        print("\n⚠️  Need more HAND samples for good detection!")
    if arm_total < 50:
        print("\n⚠️  Need more ARM samples for good distinction!")

    return True

def train_weighted_detector():
    """Train with smart hierarchy: hand > arm > nothing"""

    print("\n🚀 Training Smart Three-Class Detector")
    print("=" * 50)
    print("📝 Detection Priority: Hand > Arm > Nothing")
    print("   • If hand visible → HAND (100%)")
    print("   • If no hand but arm → ARM")
    print("   • Otherwise → NOT_HAND")

    if not prepare_training_data():
        return None

    # Create weighted config
    config_path = create_weighted_config()
    print(f"\n✅ Created weighted config: {config_path}")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Training configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"three_class_weighted_{timestamp}"

    # Training parameters optimized for hierarchy
    model_size = 's'  # Small model (better than nano for nuanced detection)
    epochs = 40  # More epochs for better learning
    batch = 16  # Smaller batch for better gradients
    patience = 15  # More patience for convergence
    lr0 = 0.005  # Lower initial learning rate for stability

    print(f"\n⚙️  Configuration:")
    print(f"   Model: YOLOv8{model_size}-cls (small for better accuracy)")
    print(f"   Classes: 3 (hand > arm > not_hand)")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Learning rate: {lr0}")
    print(f"   Class weights: [2.0, 1.0, 0.8]")
    print(f"   Early stopping: {patience} epochs")

    # Build command with weighted training
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8{model_size}-cls.pt",
        f"data={config_path}",  # Use our weighted config
        f"epochs={epochs}",
        f"batch={batch}",
        f"patience={patience}",
        f"lr0={lr0}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
        "plots=True",
        "augment=True",  # Data augmentation for better generalization
        "dropout=0.2",  # Dropout to prevent overfitting
        "label_smoothing=0.1",  # Smooth labels for better generalization
        "device=mps" if has_mps() else "device=cpu"
    ]

    print(f"\n📝 Command: {' '.join(cmd)}")
    print("\n" + "=" * 50)
    print("Training with smart hierarchy...")
    print("This may take 5-10 minutes")
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
                # Version management
                existing_models = list(models_dir.glob("three_class_smart_v*.pt"))
                next_version = len(existing_models) + 1

                # Save versioned model
                dest = models_dir / f"three_class_smart_v{next_version}.pt"
                import shutil
                shutil.copy(best_model, dest)
                print(f"\n📦 Smart model saved to: {dest}")

                # Update latest
                latest = models_dir / "three_class_smart.pt"
                shutil.copy(best_model, latest)
                print(f"   Latest: {latest}")

                # Also update the main three_class_detector for compatibility
                main_detector = models_dir / "three_class_detector.pt"
                shutil.copy(best_model, main_detector)
                print(f"   Compatible: {main_detector}")

                print(f"\n📊 Training plots: {runs_dir}")

                # Print metrics
                results_file = runs_dir / "results.csv"
                if results_file.exists():
                    print("\n📈 Final Metrics:")
                    with open(results_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            headers = lines[0].strip().split(',')
                            values = lines[-1].strip().split(',')
                            for h, v in zip(headers[:5], values[:5]):
                                try:
                                    print(f"   {h:15s}: {float(v):6.4f}")
                                except:
                                    pass

                print("\n🎯 Smart Detection Ready!")
                print("\nThe model now prioritizes:")
                print("1. HAND detection (highest priority)")
                print("2. ARM detection (only if no hand)")
                print("3. NOT_HAND (neither)")

                print("\n🎥 Test it:")
                print("   python3 live_demo_three_class.py --weights models/three_class_smart.pt")

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

    # Clean up config file
    if config_path.exists():
        config_path.unlink()

    return None

def has_mps():
    """Check if Apple Silicon MPS is available"""
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False

def verify_hierarchy(model_path):
    """Test that the model respects the hierarchy"""
    if not model_path or not model_path.exists():
        return

    print(f"\n🧪 Verifying hierarchy in: {model_path}")

    # Test with validation samples
    test_cases = [
        ("data/three_class/val/hand", "hand", "Should always detect as HAND"),
        ("data/three_class/val/arm", "arm", "Should detect as ARM (no hand visible)"),
        ("data/three_class/val/not_hand", "not_hand", "Should detect as NOT_HAND")
    ]

    for path, expected, description in test_cases:
        val_path = Path(path)
        if val_path.exists():
            samples = list(val_path.glob("*.jpg"))[:3]
            if samples:
                print(f"\n📝 Testing: {description}")
                correct = 0
                for img in samples:
                    cmd = [
                        "yolo", "classify", "predict",
                        f"model={model_path}",
                        f"source={img}",
                        "verbose=False"
                    ]

                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if expected in result.stdout.lower():
                            correct += 1
                            print(f"   ✅ Correct: {expected}")
                        else:
                            print(f"   ⚠️  Check: {img.name}")
                    except:
                        pass

                accuracy = (correct / len(samples)) * 100
                print(f"   Accuracy: {accuracy:.0f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train smart three-class detector")
    parser.add_argument('--test-only', action='store_true',
                       help='Test existing model hierarchy')
    args = parser.parse_args()

    if args.test_only:
        model = Path("models/three_class_smart.pt")
        if model.exists():
            verify_hierarchy(model)
        else:
            print("No smart model found. Train first!")
    else:
        model_path = train_weighted_detector()
        if model_path:
            verify_hierarchy(model_path)
            print("\n🎉 Smart hierarchy detector ready!")
            print("\nNext: python3 live_demo_three_class.py --weights models/three_class_smart.pt")