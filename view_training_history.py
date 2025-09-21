#!/usr/bin/env python3
"""
View and compare training metrics across model versions
"""

import json
from pathlib import Path
from typing import Dict, List
import sys

def load_metrics(metrics_dir: Path) -> Dict:
    """Load all metrics files"""
    summary_file = metrics_dir / "training_summary.json"

    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return {'models': {}}

def display_comparison(summary: Dict):
    """Display a comparison of all model versions"""
    models = summary.get('models', {})

    if not models:
        print("No training metrics found yet.")
        return

    print("\n" + "="*80)
    print("📊 TRAINING HISTORY COMPARISON")
    print("="*80)

    # Sort by version number
    sorted_models = sorted(models.items(), key=lambda x: int(x[0].replace('unified_v', '')))

    print(f"\n{'Version':<12} {'Date':<20} {'Epochs':<8} {'Final Loss':<12} {'Final Acc':<12} {'Best Acc':<12}")
    print("-"*80)

    for version, data in sorted_models:
        date = data['date'][:19]  # Trim to datetime
        epochs = data['epochs']
        final_loss = data.get('final_loss', 'N/A')
        final_acc = data.get('final_accuracy', 'N/A')
        best_acc = data.get('best_val_accuracy', 'N/A')

        # Format numbers
        if isinstance(final_loss, float):
            final_loss = f"{final_loss:.4f}"
        if isinstance(final_acc, float):
            final_acc = f"{final_acc:.1%}"
        if isinstance(best_acc, float):
            best_acc = f"{best_acc:.1%}"

        print(f"{version:<12} {date:<20} {epochs:<8} {final_loss:<12} {final_acc:<12} {best_acc:<12}")

    # Find best performing model
    best_model = None
    best_accuracy = 0
    for version, data in models.items():
        acc = data.get('best_val_accuracy', 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = version

    if best_model:
        print(f"\n🏆 Best Model: {best_model} with {best_accuracy:.1%} validation accuracy")

def view_detailed_metrics(version: str, metrics_dir: Path):
    """View detailed epoch-by-epoch metrics for a specific version"""
    metrics_file = metrics_dir / f"{version}_metrics.json"

    if not metrics_file.exists():
        print(f"Metrics file not found for {version}")
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    print(f"\n📈 Detailed Metrics for {version}")
    print("="*60)
    print(f"Training Date: {data['training_date'][:19]}")
    print(f"Configuration:")
    config = data['configuration']
    print(f"  Model: {config['model']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Corrections Added: {config.get('corrections_added', 0)}")

    print(f"\nEpoch-by-Epoch Performance:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Accuracy':<12}")
    print("-"*32)

    for epoch in data['epoch_metrics']:
        epoch_num = epoch['epoch']
        loss = epoch['train_loss']
        acc = epoch.get('val_accuracy', 'N/A')

        if isinstance(acc, float):
            acc = f"{acc:.1%}"

        print(f"{epoch_num:<8} {loss:<12.4f} {acc:<12}")

    summary = data['summary']
    print(f"\nSummary:")
    print(f"  Best Validation Accuracy: {summary['best_val_accuracy']:.1%}")
    print(f"  Final Training Loss: {summary['final_train_loss']:.4f}")
    if summary['final_val_accuracy']:
        print(f"  Final Validation Accuracy: {summary['final_val_accuracy']:.1%}")

def main():
    """Main function"""
    metrics_dir = Path("models/metrics")

    if not metrics_dir.exists():
        print("No metrics directory found. Train a model first!")
        return

    # Load summary
    summary = load_metrics(metrics_dir)

    # Display comparison
    display_comparison(summary)

    # Ask if user wants to see detailed metrics
    if summary['models']:
        print("\n" + "-"*60)
        print("Enter a version number to see detailed metrics (e.g., 'unified_v5')")
        print("Or press Enter to exit: ", end='')

        version = input().strip()
        if version:
            if not version.startswith('unified_v'):
                version = f'unified_v{version}'
            view_detailed_metrics(version, metrics_dir)

if __name__ == "__main__":
    main()