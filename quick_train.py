#!/usr/bin/env python3
"""
🚀 Quick Training with Corrections (10 epochs)
"""

import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import os

def incorporate_corrections(data_path):
    """Add correction images to training data"""
    corrections_dir = Path("../unified-detector-client/corrections")

    if not corrections_dir.exists():
        print("📝 No corrections found")
        return 0

    print("\n🔄 Incorporating corrections...")
    total = 0

    for class_name in ['hand', 'arm', 'not_hand']:
        src_dir = corrections_dir / class_name
        if not src_dir.exists() or not any(src_dir.iterdir()):
            continue

        train_dir = data_path / 'train' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)

        for img in src_dir.glob("*.jpg"):
            dest = train_dir / f"CORRECTED_{img.name}"
            shutil.copy2(img, dest)
            total += 1
            print(f"   Added: {img.name} → {class_name}")

    print(f"\n✅ Incorporated {total} corrections")
    return total

def quick_train():
    """Quick 10-epoch training"""
    print("\n" + "="*60)
    print("🚀 QUICK TRAINING WITH CORRECTIONS (10 epochs)")
    print("="*60)

    # Incorporate corrections
    data_path = Path("data/hand_cls")
    corrections = incorporate_corrections(data_path)

    # Count images
    print("\n📊 Dataset:")
    for split in ['train', 'val']:
        for cls in ['hand', 'arm', 'not_hand']:
            path = data_path / split / cls
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                print(f"  {split}/{cls}: {count} images")

    # Training params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"quick_{timestamp}"

    print(f"\n⚙️ Starting quick training...")
    print(f"   Run: {run_name}")
    print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")

    # Build command for quick training
    cmd = [
        "yolo", "classify", "train",
        f"model=yolov8s-cls.pt",
        f"data={data_path.absolute()}",
        f"epochs=10",  # Quick training
        f"batch=16",
        f"patience=5",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
        "device=mps"
    ]

    # Run training
    process = subprocess.run(cmd)

    if process.returncode == 0:
        print("\n✅ Training completed!")

        # Find and save model
        runs_dir = Path("runs/classify") / run_name
        best_model = runs_dir / "weights/best.pt"

        if best_model.exists():
            # Version management
            models_dir = Path("models")
            existing = list(models_dir.glob("unified_v*.pt"))
            next_version = len(existing) + 1

            # Save versioned
            dest = models_dir / f"unified_v{next_version}.pt"
            shutil.copy(best_model, dest)
            print(f"\n📦 Saved: {dest}")

            # Update latest
            latest = models_dir / "unified_detector.pt"
            shutil.copy(best_model, latest)

            # Deploy to client
            client_model = Path("../unified-detector-client/weights/model.pt")
            client_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_model, client_model)
            print(f"   Deployed to client")

            # Restart server
            print("\n🔄 Restarting server...")
            os.system("pkill -f 'python3 local-server.py' 2>/dev/null")
            import time
            time.sleep(1)
            subprocess.Popen(['python3', '../unified-detector-client/local-server.py'])

            print("\n🎉 SUCCESS! Model retrained and deployed!")
            print("   Test at: http://localhost:3002/detect")

            return dest
    else:
        print("\n❌ Training failed")
        return None

if __name__ == "__main__":
    quick_train()