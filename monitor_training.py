#!/usr/bin/env python3
"""
🚀 Live Training Monitor for Hand Detection Model
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess
import psutil

class TrainingMonitor:
    def __init__(self):
        self.runs_dir = Path("runs/classify")
        self.models_dir = Path("models")
        self.data_dir = Path("data/hand_cls")
        self.corrections_dir = Path("../unified-detector-client/corrections")

    def check_gpu_status(self):
        """Check if MPS/CUDA is available"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "MPS (Apple Silicon)", True
            elif torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})", True
            else:
                return "CPU", False
        except:
            return "CPU", False

    def count_dataset(self):
        """Count images in dataset"""
        stats = {
            'train': {'hand': 0, 'arm': 0, 'not_hand': 0},
            'val': {'hand': 0, 'arm': 0, 'not_hand': 0}
        }

        for split in ['train', 'val']:
            for cls in ['hand', 'arm', 'not_hand']:
                path = self.data_dir / split / cls
                if path.exists():
                    stats[split][cls] = len(list(path.glob("*.jpg")))

        # Count corrections
        corrections = 0
        if self.corrections_dir.exists():
            for cls in ['hand', 'arm', 'not_hand']:
                cls_dir = self.corrections_dir / cls
                if cls_dir.exists():
                    corrections += len(list(cls_dir.glob("*.jpg")))

        return stats, corrections

    def find_active_training(self):
        """Find if training is currently running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'yolo' in str(proc.info['cmdline']).lower():
                    return proc.info['pid']
            except:
                pass
        return None

    def get_latest_run(self):
        """Get the latest training run"""
        if not self.runs_dir.exists():
            return None

        runs = sorted([d for d in self.runs_dir.iterdir() if d.is_dir()],
                     key=lambda x: x.stat().st_mtime, reverse=True)

        if runs:
            return runs[0]
        return None

    def parse_training_log(self, run_path):
        """Parse training results from YOLO output"""
        results_file = run_path / "results.csv"
        if not results_file.exists():
            return None

        with open(results_file) as f:
            lines = f.readlines()
            if len(lines) > 1:
                # Get last line (latest epoch)
                last_line = lines[-1].strip().split(',')
                headers = lines[0].strip().split(',')

                data = {}
                for i, header in enumerate(headers):
                    if i < len(last_line):
                        try:
                            data[header.strip()] = float(last_line[i])
                        except:
                            data[header.strip()] = last_line[i]

                return data
        return None

    def display_status(self):
        """Display current training status"""
        os.system('clear')

        print("=" * 60)
        print("🚀 HAND DETECTION TRAINING MONITOR")
        print("=" * 60)

        # GPU Status
        device, has_gpu = self.check_gpu_status()
        status_emoji = "✅" if has_gpu else "⚠️"
        print(f"\n📊 Hardware: {status_emoji} {device}")

        # Dataset Stats
        stats, corrections = self.count_dataset()
        print(f"\n📁 Dataset Statistics:")
        print(f"   Training:")
        for cls in ['hand', 'arm', 'not_hand']:
            count = stats['train'][cls]
            print(f"     {cls:10s}: {count:4d} images")
        print(f"   Validation:")
        for cls in ['hand', 'arm', 'not_hand']:
            count = stats['val'][cls]
            print(f"     {cls:10s}: {count:4d} images")

        if corrections > 0:
            print(f"\n   📝 Pending Corrections: {corrections} images")

        # Training Status
        training_pid = self.find_active_training()
        if training_pid:
            print(f"\n⏳ TRAINING IN PROGRESS (PID: {training_pid})")

            # Get latest run info
            latest_run = self.get_latest_run()
            if latest_run:
                results = self.parse_training_log(latest_run)
                if results:
                    epoch = int(results.get('epoch', 0))
                    train_loss = results.get('train/loss', 0)
                    val_loss = results.get('val/loss', 0)
                    accuracy = results.get('metrics/accuracy_top1', 0)

                    print(f"\n   Epoch: {epoch}")
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Accuracy: {accuracy:.1%}")

                    # Check for overfitting
                    if val_loss > train_loss * 1.5:
                        print(f"\n   ⚠️ WARNING: Possible overfitting detected!")
        else:
            print(f"\n✅ No active training")

            # Show latest model
            models = sorted(self.models_dir.glob("unified_v*.pt"),
                          key=lambda x: x.stat().st_mtime, reverse=True)
            if models:
                latest = models[0]
                mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
                print(f"\n   Latest Model: {latest.name}")
                print(f"   Trained: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Check if deployed
                client_model = Path("../unified-detector-client/weights/model.pt")
                if client_model.exists():
                    if client_model.stat().st_mtime >= latest.stat().st_mtime:
                        print(f"   Status: ✅ Deployed to client")
                    else:
                        print(f"   Status: ⚠️ Client using older model")

        # Server Status
        print(f"\n🖥️ Server Status:")
        try:
            result = subprocess.run(['lsof', '-i', ':8000'],
                                  capture_output=True, text=True)
            if 'LISTEN' in result.stdout:
                print(f"   ✅ Detection server running on port 8000")
            else:
                print(f"   ❌ Detection server not running")
        except:
            print(f"   ⚠️ Could not check server status")

        print("\n" + "=" * 60)
        print("Commands:")
        print("  [T] Start training")
        print("  [C] Incorporate corrections")
        print("  [D] Deploy latest model")
        print("  [S] Restart server")
        print("  [Q] Quit")
        print("=" * 60)

    def start_training(self):
        """Start training process"""
        print("\n🚀 Starting training...")
        subprocess.Popen(['python3', 'train_unified.py'])
        time.sleep(2)

    def incorporate_corrections(self):
        """Run training with corrections"""
        print("\n📝 Training with corrections...")
        subprocess.run(['python3', 'train_unified.py'])

    def deploy_model(self):
        """Deploy latest model to client"""
        models = sorted(self.models_dir.glob("unified_v*.pt"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if models:
            latest = models[0]
            dest = Path("../unified-detector-client/weights/model.pt")
            dest.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(latest, dest)
            print(f"\n✅ Deployed {latest.name} to client")
        else:
            print("\n❌ No model found to deploy")

    def restart_server(self):
        """Restart detection server"""
        print("\n🔄 Restarting server...")
        os.system("pkill -f 'python3 local-server.py' 2>/dev/null")
        time.sleep(1)
        subprocess.Popen(['python3', '../unified-detector-client/local-server.py'])
        print("✅ Server restarted")

    def run(self):
        """Main monitoring loop"""
        while True:
            self.display_status()

            # Check for input (non-blocking)
            try:
                import select
                import sys

                # Check if input is available
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    cmd = input().strip().upper()

                    if cmd == 'T':
                        self.start_training()
                    elif cmd == 'C':
                        self.incorporate_corrections()
                    elif cmd == 'D':
                        self.deploy_model()
                    elif cmd == 'S':
                        self.restart_server()
                    elif cmd == 'Q':
                        break
            except:
                pass

            time.sleep(2)  # Update every 2 seconds

if __name__ == "__main__":
    monitor = TrainingMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped")