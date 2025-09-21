#!/usr/bin/env python3
"""
🚀 Train with Live Monitoring
"""

import subprocess
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import threading

class LiveTrainingMonitor:
    def __init__(self):
        self.training_process = None
        self.monitoring = False
        self.runs_dir = Path("runs/classify")

    def parse_yolo_output(self, line):
        """Parse YOLO training output for metrics"""
        if 'train:' in line.lower():
            # Extract metrics from training output
            parts = line.split()
            metrics = {}

            for i, part in enumerate(parts):
                if 'loss' in part.lower():
                    try:
                        metrics['loss'] = float(parts[i+1])
                    except:
                        pass
                elif 'acc' in part.lower():
                    try:
                        metrics['accuracy'] = float(parts[i+1])
                    except:
                        pass

            return metrics
        return None

    def monitor_training(self):
        """Monitor training output in real-time"""
        while self.monitoring and self.training_process:
            line = self.training_process.stdout.readline()
            if line:
                line = line.decode('utf-8').strip()

                # Display all output
                print(line)

                # Parse metrics
                metrics = self.parse_yolo_output(line)
                if metrics:
                    self.display_metrics(metrics)

                # Check for completion
                if 'training completed' in line.lower() or 'best model saved' in line.lower():
                    print("\n✅ Training completed successfully!")
                    self.monitoring = False

            else:
                if self.training_process.poll() is not None:
                    # Process has finished
                    self.monitoring = False
                    break

            time.sleep(0.1)

    def display_metrics(self, metrics):
        """Display metrics in a formatted way"""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"📊 {metric_str}")

    def start_training(self, quick=False):
        """Start training with monitoring"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING WITH LIVE MONITORING")
        print("="*60)

        # Check for corrections
        corrections_dir = Path("../unified-detector-client/corrections")
        correction_count = 0
        if corrections_dir.exists():
            for cls_dir in corrections_dir.glob("*"):
                if cls_dir.is_dir():
                    correction_count += len(list(cls_dir.glob("*.jpg")))

        if correction_count > 0:
            print(f"\n📝 Found {correction_count} corrections to incorporate")

        # Build training command
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"unified_{timestamp}"

        # Training parameters
        epochs = 10 if quick else 50
        batch = 16

        print(f"\n⚙️ Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch}")
        print(f"   Run Name: {run_name}")
        print(f"   Device: MPS (Apple Silicon)")

        # Start training process
        print(f"\n📊 Starting training at {datetime.now().strftime('%H:%M:%S')}...")
        print("-" * 60)

        # Run the unified training script
        self.training_process = subprocess.Popen(
            ['python3', 'train_unified.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )

        # Start monitoring in separate thread
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_training)
        monitor_thread.start()

        # Wait for completion
        monitor_thread.join()

        if self.training_process.returncode == 0:
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)

            # Offer to deploy
            response = input("\n🚀 Deploy to client? (y/n): ")
            if response.lower() == 'y':
                self.deploy_model()

        else:
            print("\n❌ Training failed")

    def deploy_model(self):
        """Deploy model and restart server"""
        print("\n📦 Deploying model to client...")

        # Copy model
        models_dir = Path("models")
        latest = models_dir / "unified_detector.pt"

        if latest.exists():
            dest = Path("../unified-detector-client/weights/model.pt")
            dest.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(latest, dest)
            print(f"✅ Model copied to client")

            # Restart server
            print("\n🔄 Restarting server...")
            os.system("pkill -f 'python3 local-server.py' 2>/dev/null")
            time.sleep(1)
            subprocess.Popen(['python3', '../unified-detector-client/local-server.py'])
            print("✅ Server restarted with new model")

            print(f"\n🎉 Deployment complete!")
            print(f"   Test at: http://localhost:3000")
        else:
            print("❌ Model not found")

def main():
    monitor = LiveTrainingMonitor()

    print("🚀 ML Training Pipeline")
    print("\nOptions:")
    print("  1. Quick training (10 epochs)")
    print("  2. Full training (50 epochs)")
    print("  3. Deploy existing model")
    print("  4. Exit")

    choice = input("\nSelect option [1-4]: ")

    if choice == '1':
        monitor.start_training(quick=True)
    elif choice == '2':
        monitor.start_training(quick=False)
    elif choice == '3':
        monitor.deploy_model()
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()