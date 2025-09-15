#!/usr/bin/env python3
"""
Hand Detection Training Pipeline
Phase 1: Train to detect hands
Phase 2: Train to recognize gestures
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

class HandDetectorTrainer:
    def __init__(self):
        self.project_root = Path.cwd()
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Create scratchpad for tracking experiments
        self.scratchpad = self.project_root / "claude.scratchpad.md"

    def prepare_yolo_dataset(self, phase="hand_detection"):
        """Prepare dataset in YOLO format"""
        print(f"\n📦 Preparing dataset for {phase}...")

        if phase == "hand_detection":
            # For phase 1: Binary classification (hand vs no hand)
            classes = ['hand']
        else:
            # For phase 2: Multiple gesture classes
            raw_dir = self.data_dir / "raw"
            if raw_dir.exists():
                classes = [d.name for d in raw_dir.iterdir() if d.is_dir() and d.name != 'none']
            else:
                classes = ['ok', 'thumbs_up', 'peace', 'fist', 'point']

        # Create YOLO directory structure
        yolo_dir = self.data_dir / "yolo" / phase
        yolo_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset.yaml
        yaml_content = f"""# {phase} dataset
path: {yolo_dir}
train: images/train
val: images/val

nc: {len(classes)}
names: {classes}
"""
        yaml_path = yolo_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)

        print(f"  ✅ Created dataset.yaml with {len(classes)} classes")
        return yolo_dir, yaml_path

    def train_hand_detector(self, epochs=30, batch_size=16, model_size='n'):
        """Train YOLO for hand detection (Phase 1)"""
        print("\n🚀 Starting Hand Detection Training (Phase 1)")
        print("="*50)

        # Prepare dataset
        yolo_dir, yaml_path = self.prepare_yolo_dataset("hand_detection")

        # Check if we have data
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            print("\n❌ No training data found!")
            print("   Run: python collect_data.py")
            print("   Collect at least 100 images with hands and 50 without")
            return None

        # Create training command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"hand_detect_{timestamp}"

        cmd = [
            "yolo", "detect", "train",
            f"model=yolov8{model_size}.pt",
            f"data={yaml_path}",
            f"epochs={epochs}",
            f"batch={batch_size}",
            f"name={run_name}",
            "device=0" if self._has_gpu() else "device=cpu",
            "patience=10",
            "save=True",
            "exist_ok=True"
        ]

        print(f"\n⏳ Training configuration:")
        print(f"   Model: YOLOv8{model_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {'GPU' if self._has_gpu() else 'CPU'}")

        # Log to scratchpad
        self._log_experiment(f"Hand Detection - {run_name}", cmd)

        # Start training
        log_file = self.models_dir / f"{run_name}.log"
        print(f"\n📝 Logging to: {log_file}")
        print("\n" + "="*50)

        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                # Stream output
                for line in process.stdout:
                    print(line.rstrip())
                    f.write(line)
                    f.flush()

                process.wait()

            if process.returncode == 0:
                print("\n✅ Training completed successfully!")
                # Find best model
                runs_dir = Path("runs/detect") / run_name / "weights"
                if runs_dir.exists():
                    best_model = runs_dir / "best.pt"
                    if best_model.exists():
                        # Copy to models directory
                        dest = self.models_dir / f"hand_detector_v1.pt"
                        shutil.copy(best_model, dest)
                        print(f"   Model saved: {dest}")
                        return dest
            else:
                print(f"\n❌ Training failed with code {process.returncode}")

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            process.terminate()

        return None

    def train_gesture_classifier(self, epochs=50, batch_size=32, model_size='n'):
        """Train for gesture recognition (Phase 2)"""
        print("\n🚀 Starting Gesture Recognition Training (Phase 2)")
        print("="*50)

        # Check for hand detector model
        hand_model = self.models_dir / "hand_detector_v1.pt"
        if not hand_model.exists():
            print("\n❌ Hand detector not found!")
            print("   Train hand detector first (Phase 1)")
            return None

        # Prepare dataset
        yolo_dir, yaml_path = self.prepare_yolo_dataset("gesture_recognition")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"gesture_{timestamp}"

        cmd = [
            "yolo", "classify", "train",
            f"model=yolov8{model_size}-cls.pt",
            f"data={self.data_dir / 'raw'}",
            f"epochs={epochs}",
            f"batch={batch_size}",
            f"name={run_name}",
            "device=0" if self._has_gpu() else "device=cpu",
            "patience=15",
            "save=True"
        ]

        print(f"\n⏳ Training configuration:")
        print(f"   Model: YOLOv8{model_size}-cls")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")

        # Log to scratchpad
        self._log_experiment(f"Gesture Recognition - {run_name}", cmd)

        # Similar training process as hand detector
        log_file = self.models_dir / f"{run_name}.log"

        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                for line in process.stdout:
                    print(line.rstrip())
                    f.write(line)
                    f.flush()

                process.wait()

            if process.returncode == 0:
                print("\n✅ Training completed successfully!")
                # Find best model
                runs_dir = Path("runs/classify") / run_name / "weights"
                if runs_dir.exists():
                    best_model = runs_dir / "best.pt"
                    if best_model.exists():
                        dest = self.models_dir / f"gesture_classifier_v1.pt"
                        shutil.copy(best_model, dest)
                        print(f"   Model saved: {dest}")
                        return dest

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted")
            process.terminate()

        return None

    def _has_gpu(self):
        """Check if GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            # Check for Apple Silicon
            import platform
            return platform.processor() == 'arm' and platform.system() == 'Darwin'

    def _log_experiment(self, name, cmd):
        """Log experiment to scratchpad"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.scratchpad, 'a') as f:
            f.write(f"\n## {name}\n")
            f.write(f"Started: {timestamp}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Train hand detection models")
    parser.add_argument('phase', choices=['hand', 'gesture', 'both'],
                       help='Training phase: hand detection, gesture recognition, or both')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--model', choices=['n', 's', 'm'], default='n',
                       help='Model size: n=nano, s=small, m=medium')
    args = parser.parse_args()

    trainer = HandDetectorTrainer()

    if args.phase in ['hand', 'both']:
        model = trainer.train_hand_detector(args.epochs, args.batch, args.model)
        if model:
            print(f"\n📦 Hand detector ready: {model}")

    if args.phase in ['gesture', 'both']:
        model = trainer.train_gesture_classifier(args.epochs, args.batch, args.model)
        if model:
            print(f"\n📦 Gesture classifier ready: {model}")

if __name__ == "__main__":
    main()