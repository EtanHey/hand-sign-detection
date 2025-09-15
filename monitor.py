#!/usr/bin/env python3
"""
Monitor system resources during training
Run this in a separate terminal while training
"""

import psutil
import time
import os
import subprocess
from datetime import datetime

def get_gpu_info():
    """Get GPU usage info"""
    # For NVIDIA GPUs
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return f"GPU: {values[0]}% | VRAM: {values[1]}/{values[2]}MB"
    except:
        pass

    # For Apple Silicon
    try:
        result = subprocess.run(['powermetrics', '--samplers', 'gpu_power', '-i', '1', '-n', '1'],
                              capture_output=True, text=True)
        if "GPU" in result.stdout:
            return "GPU: Active (Apple Silicon)"
    except:
        pass

    return "GPU: N/A"

def monitor_resources():
    """Monitor CPU, RAM, and GPU usage"""
    print("📊 System Resource Monitor")
    print("Press Ctrl+C to stop\n")
    print("Time     | CPU % | RAM % | RAM Used  | " + ("GPU Info" if get_gpu_info() != "GPU: N/A" else ""))
    print("-" * 70)

    try:
        while True:
            # Get metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            ram_used = memory.used / (1024**3)  # Convert to GB
            ram_total = memory.total / (1024**3)

            # Format time
            current_time = datetime.now().strftime("%H:%M:%S")

            # Build output line
            output = f"{current_time} | {cpu_percent:5.1f} | {ram_percent:5.1f} | {ram_used:4.1f}/{ram_total:4.1f}GB"

            # Add GPU info if available
            gpu_info = get_gpu_info()
            if gpu_info != "GPU: N/A":
                output += f" | {gpu_info}"

            print(output)

            # Also check for Python processes using lots of CPU
            python_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if 'python' in proc.info['name'].lower():
                    cpu = proc.info['cpu_percent']
                    if cpu > 10:  # Only show if using >10% CPU
                        python_procs.append(f"PID {proc.info['pid']}: {cpu:.1f}%")

            if python_procs:
                print(f"         | Python processes: {', '.join(python_procs)}")

            time.sleep(2)  # Update every 2 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def check_training_logs():
    """Check latest training log"""
    runs_dir = Path("runs/classify")
    if runs_dir.exists():
        # Find most recent run
        runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()],
                     key=lambda x: x.stat().st_mtime, reverse=True)
        if runs:
            latest = runs[0]
            results_file = latest / "results.csv"
            if results_file.exists():
                print(f"\n📈 Training metrics from: {latest.name}")
                # Show last few lines of results
                with open(results_file) as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        headers = lines[0].strip().split(',')
                        last_line = lines[-1].strip().split(',')
                        for h, v in zip(headers[:5], last_line[:5]):
                            print(f"  {h}: {v}")

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1 and sys.argv[1] == "logs":
        check_training_logs()
    else:
        monitor_resources()