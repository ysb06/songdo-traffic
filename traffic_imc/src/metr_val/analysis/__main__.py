#!/usr/bin/env python3
"""
Complete RNN Model Performance Analysis Script

This script provides a comprehensive analysis of your trained RNN model's performance
including prediction accuracy, error patterns, temporal analysis, and training diagnostics.

Usage:
    python analyze_model_performance.py --model_path ./output/rnn/best-epoch=14-val_loss=0.00.ckpt
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from metr_val.analysis.visualization_analysis import analyze_rnn_predictions
from metr_val.analysis.training_diagnostics import analyze_training_progress


def main():
    parser = argparse.ArgumentParser(description='Comprehensive RNN Model Performance Analysis')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.ckpt file)')
    
    # Optional arguments
    parser.add_argument('--data_path', type=str, 
                       default='./data/selected_small_v1/metr-imc.h5',
                       help='Path to data file (default: ./data/selected_small_v1/metr-imc.h5)')
    
    parser.add_argument('--results_dir', type=str, 
                       default='./analysis_results',
                       help='Directory to save analysis results (default: ./analysis_results)')
    
    parser.add_argument('--training_logs', type=str,
                       default='./wandb',
                       help='Path to training logs directory (default: ./wandb)')
    
    parser.add_argument('--skip_predictions', action='store_true',
                       help='Skip prediction analysis (only run training diagnostics)')
    
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training diagnostics (only run prediction analysis)')
    
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for time series plots (default: 1000)')
    
    args = parser.parse_args()
    
    # Validate inputs
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("RNN Model Performance Analysis")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Results will be saved to: {args.results_dir}")
    print("=" * 80)
    
    # Run prediction analysis
    if not args.skip_predictions:
        print("\n🔍 STARTING PREDICTION ANALYSIS...")
        analyzer = analyze_rnn_predictions(
            model_checkpoint_path=str(model_path),
            data_path=str(data_path),
            results_dir=args.results_dir
        )
        print("✅ Prediction analysis completed!")
    
    # Run training diagnostics
    if not args.skip_training:
        print("\n📈 STARTING TRAINING DIAGNOSTICS...")
        try:
            convergence_info = analyze_training_progress(
                log_path=args.training_logs,
                results_dir=args.results_dir
            )
            print("✅ Training diagnostics completed!")
        except FileNotFoundError as e:
            print(f"⚠️  Training diagnostics skipped: {e}")
            print("   Training logs not found. This is normal if you don't have WandB logs.")
    
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"📁 Results saved to: {Path(args.results_dir).absolute()}")
    print("\nGenerated files:")
    print("📊 Static plots:")
    print("   - plots/time_series_comparison.png")
    print("   - plots/error_analysis.png")
    print("   - plots/temporal_analysis.png")
    print("   - plots/convergence_analysis.png (if training logs available)")
    print("   - plots/training_curves.png (if training logs available)")
    print("\n🌐 Interactive dashboards:")
    print("   - interactive/dashboard.html")
    print("   - interactive/training_dashboard.html (if training logs available)")
    print("\n💾 Data files:")
    print("   - data/predictions.pkl")
    print("\n📖 How to use results:")
    print("   1. Open interactive HTML files in your web browser")
    print("   2. View static PNG plots for reports/presentations")
    print("   3. Load predictions.pkl for custom analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()