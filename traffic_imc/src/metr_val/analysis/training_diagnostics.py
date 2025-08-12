"""
Training diagnostics and model performance analysis
Analyzes training curves, convergence patterns, and prediction intervals
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TrainingDiagnostics:
    """Analyze training progress and model convergence"""
    
    def __init__(self, results_dir: str = "./output/analysis_results"):
        """
        Initialize training diagnostics
        
        Args:
            results_dir: Directory to save diagnostic results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "interactive").mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
    def load_training_logs(self, log_path: str) -> pd.DataFrame:
        """
        Load training logs from CSV or WandB format
        
        Args:
            log_path: Path to training log file
            
        Returns:
            DataFrame with training metrics
        """
        log_path = Path(log_path)
        
        if log_path.suffix == '.csv':
            # Lightning CSV logger format
            df = pd.read_csv(log_path)
        else:
            # Try to find CSV files in the directory
            csv_files = list(log_path.glob("**/*.csv"))
            if csv_files:
                # Use the first CSV file found
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError(f"No CSV training logs found in {log_path}")
        
        return df
    
    def plot_training_curves(self, log_data: pd.DataFrame, save_name: str = "training_curves"):
        """
        Plot training and validation curves
        
        Args:
            log_data: DataFrame with training metrics
            save_name: Name for saved plot file
        """
        print("Creating training curves plot...")
        
        # Identify available metrics
        metric_columns = [col for col in log_data.columns if any(metric in col.lower() 
                         for metric in ['loss', 'mae', 'rmse', 'mse'])]
        
        # Separate train and validation metrics
        train_metrics = [col for col in metric_columns if 'train' in col.lower()]
        val_metrics = [col for col in metric_columns if 'val' in col.lower()]
        
        # Create subplots based on available metrics
        n_metrics = max(len(train_metrics), len(val_metrics))
        if n_metrics == 0:
            print("No training metrics found in log data")
            return
            
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
            
        fig.suptitle('Training Progress: Loss and Metrics Over Time', fontsize=16)
        
        # Plot each metric
        for i, (train_col, val_col) in enumerate(zip(train_metrics, val_metrics)):
            ax = axes[i] if i < len(axes) else axes[-1]
            
            # Plot training metric
            if train_col in log_data.columns:
                train_data = log_data[train_col].dropna()
                if len(train_data) > 0:
                    ax.plot(train_data.index, train_data.values, 
                           label=f'Training {train_col}', alpha=0.8, linewidth=2)
            
            # Plot validation metric
            if val_col in log_data.columns:
                val_data = log_data[val_col].dropna()
                if len(val_data) > 0:
                    ax.plot(val_data.index, val_data.values, 
                           label=f'Validation {val_col}', alpha=0.8, linewidth=2)
            
            ax.set_title(f'{train_col.replace("train_", "").replace("val_", "").upper()} Progress')
            ax.set_xlabel('Epoch/Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add smoothed trend line
            if val_col in log_data.columns:
                val_data = log_data[val_col].dropna()
                if len(val_data) > 10:  # Only if enough data points
                    window = max(1, len(val_data) // 10)
                    smoothed = val_data.rolling(window=window, center=True).mean()
                    ax.plot(smoothed.index, smoothed.values, 
                           '--', alpha=0.6, label=f'Smoothed {val_col}')
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_convergence(self, log_data: pd.DataFrame) -> Dict:
        """
        Analyze model convergence patterns
        
        Args:
            log_data: DataFrame with training metrics
            
        Returns:
            Dictionary with convergence analysis results
        """
        print("Analyzing convergence patterns...")
        
        analysis = {}
        
        # Find validation loss column
        val_loss_cols = [col for col in log_data.columns if 'val' in col.lower() and 'loss' in col.lower()]
        if not val_loss_cols:
            print("No validation loss found for convergence analysis")
            return analysis
            
        val_loss_col = val_loss_cols[0]
        val_loss = log_data[val_loss_col].dropna()
        
        if len(val_loss) < 5:
            print("Insufficient data for convergence analysis")
            return analysis
        
        # Find best epoch and early stopping point
        best_epoch = val_loss.idxmin()
        best_loss = val_loss.min()
        
        # Detect early stopping (look for consecutive increases)
        early_stop_detected = False
        patience_count = 0
        patience_threshold = 5  # Typical early stopping patience
        
        for i in range(best_epoch + 1, len(val_loss)):
            if val_loss.iloc[i] > best_loss:
                patience_count += 1
                if patience_count >= patience_threshold:
                    early_stop_detected = True
                    break
            else:
                patience_count = 0
                best_loss = val_loss.iloc[i]
                best_epoch = i
        
        # Calculate convergence metrics
        initial_loss = val_loss.iloc[0] if len(val_loss) > 0 else None
        final_loss = val_loss.iloc[-1]
        improvement = initial_loss - final_loss if initial_loss else None
        improvement_pct = (improvement / initial_loss * 100) if initial_loss and initial_loss != 0 else None
        
        # Detect overfitting (validation loss increases while training loss decreases)
        train_loss_cols = [col for col in log_data.columns if 'train' in col.lower() and 'loss' in col.lower()]
        overfitting_detected = False
        
        if train_loss_cols:
            train_loss = log_data[train_loss_cols[0]].dropna()
            if len(train_loss) >= len(val_loss):
                # Compare last 10% of training
                tail_length = max(1, len(val_loss) // 10)
                val_trend = val_loss.tail(tail_length).diff().mean()
                train_trend = train_loss.tail(tail_length).diff().mean()
                
                if val_trend > 0 and train_trend < 0:  # Val loss increasing, train loss decreasing
                    overfitting_detected = True
        
        # Learning rate analysis (if available)
        lr_cols = [col for col in log_data.columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        lr_analysis = {}
        if lr_cols:
            lr_data = log_data[lr_cols[0]].dropna()
            lr_analysis = {
                'initial_lr': lr_data.iloc[0] if len(lr_data) > 0 else None,
                'final_lr': lr_data.iloc[-1] if len(lr_data) > 0 else None,
                'lr_reductions': len([i for i in range(1, len(lr_data)) if lr_data.iloc[i] < lr_data.iloc[i-1]]),
                'min_lr': lr_data.min(),
                'max_lr': lr_data.max()
            }
        
        analysis = {
            'best_epoch': int(best_epoch),
            'best_val_loss': float(best_loss),
            'final_val_loss': float(final_loss),
            'initial_val_loss': float(initial_loss) if initial_loss else None,
            'total_improvement': float(improvement) if improvement else None,
            'improvement_percentage': float(improvement_pct) if improvement_pct else None,
            'early_stopping_detected': early_stop_detected,
            'overfitting_detected': overfitting_detected,
            'total_epochs': len(val_loss),
            'learning_rate_analysis': lr_analysis
        }
        
        return analysis
    
    def plot_convergence_analysis(self, log_data: pd.DataFrame, convergence_info: Dict):
        """
        Create detailed convergence analysis plots
        
        Args:
            log_data: DataFrame with training metrics
            convergence_info: Results from analyze_convergence
        """
        print("Creating convergence analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Convergence Analysis', fontsize=16)
        
        # Find loss columns
        val_loss_cols = [col for col in log_data.columns if 'val' in col.lower() and 'loss' in col.lower()]
        train_loss_cols = [col for col in log_data.columns if 'train' in col.lower() and 'loss' in col.lower()]
        
        if val_loss_cols:
            val_loss = log_data[val_loss_cols[0]].dropna()
            
            # 1. Loss curve with best epoch marked
            axes[0, 0].plot(val_loss.index, val_loss.values, label='Validation Loss', linewidth=2)
            if train_loss_cols:
                train_loss = log_data[train_loss_cols[0]].dropna()
                axes[0, 0].plot(train_loss.index, train_loss.values, label='Training Loss', linewidth=2)
            
            # Mark best epoch
            best_epoch = convergence_info.get('best_epoch', 0)
            best_loss = convergence_info.get('best_val_loss', 0)
            axes[0, 0].scatter([best_epoch], [best_loss], color='red', s=100, zorder=5, 
                             label=f'Best Epoch ({best_epoch})')
            axes[0, 0].set_title('Training Progress with Best Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Loss improvement over time
            initial_loss = convergence_info.get('initial_val_loss', val_loss.iloc[0])
            improvement = initial_loss - val_loss if initial_loss else val_loss
            axes[0, 1].plot(improvement.index, improvement.values, color='green', linewidth=2)
            axes[0, 1].set_title('Cumulative Loss Improvement')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss Improvement')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Loss derivatives (learning speed)
            if len(val_loss) > 1:
                loss_diff = val_loss.diff()
                axes[1, 0].plot(loss_diff.index[1:], loss_diff.values[1:], alpha=0.7)
                axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[1, 0].set_title('Validation Loss Change Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss Δ')
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Learning rate schedule (if available)
        lr_cols = [col for col in log_data.columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        if lr_cols:
            lr_data = log_data[lr_cols[0]].dropna()
            axes[1, 1].plot(lr_data.index, lr_data.values, color='orange', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Plot loss distribution instead
            if val_loss_cols:
                axes[1, 1].hist(val_loss.values, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Validation Loss Distribution')
                axes[1, 1].set_xlabel('Loss Value')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print convergence summary
        print("\nConvergence Analysis Summary:")
        print(f"Best epoch: {convergence_info.get('best_epoch', 'N/A')}")
        print(f"Best validation loss: {convergence_info.get('best_val_loss', 'N/A'):.6f}")
        print(f"Total improvement: {convergence_info.get('improvement_percentage', 'N/A'):.2f}%")
        print(f"Early stopping detected: {convergence_info.get('early_stopping_detected', 'N/A')}")
        print(f"Overfitting detected: {convergence_info.get('overfitting_detected', 'N/A')}")
        
        if convergence_info.get('learning_rate_analysis'):
            lr_info = convergence_info['learning_rate_analysis']
            print(f"Learning rate reductions: {lr_info.get('lr_reductions', 'N/A')}")
            print(f"Final learning rate: {lr_info.get('final_lr', 'N/A'):.2e}")
    
    def create_interactive_training_dashboard(self, log_data: pd.DataFrame):
        """
        Create interactive training progress dashboard
        
        Args:
            log_data: DataFrame with training metrics
        """
        print("Creating interactive training dashboard...")
        
        # Identify metrics
        metric_columns = [col for col in log_data.columns if any(metric in col.lower() 
                         for metric in ['loss', 'mae', 'rmse', 'mse', 'lr', 'learning_rate'])]
        
        train_metrics = [col for col in metric_columns if 'train' in col.lower()]
        val_metrics = [col for col in metric_columns if 'val' in col.lower()]
        lr_metrics = [col for col in metric_columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        
        # Create subplot structure
        subplot_titles = ['Loss Curves', 'Metrics Comparison', 'Learning Rate', 'Training Statistics']
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Loss curves
        for train_col in train_metrics:
            if 'loss' in train_col.lower():
                train_data = log_data[train_col].dropna()
                fig.add_trace(
                    go.Scatter(x=train_data.index, y=train_data.values,
                             name=f'Training Loss', line=dict(color='blue')),
                    row=1, col=1
                )
                break
        
        for val_col in val_metrics:
            if 'loss' in val_col.lower():
                val_data = log_data[val_col].dropna()
                fig.add_trace(
                    go.Scatter(x=val_data.index, y=val_data.values,
                             name=f'Validation Loss', line=dict(color='red')),
                    row=1, col=1, secondary_y=False
                )
                break
        
        # 2. Other metrics comparison
        colors = ['green', 'orange', 'purple', 'brown']
        for i, metric_col in enumerate([col for col in val_metrics if 'loss' not in col.lower()][:4]):
            metric_data = log_data[metric_col].dropna()
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(x=metric_data.index, y=metric_data.values,
                         name=metric_col.replace('val_', '').upper(), 
                         line=dict(color=color)),
                row=1, col=2
            )
        
        # 3. Learning rate
        if lr_metrics:
            lr_data = log_data[lr_metrics[0]].dropna()
            fig.add_trace(
                go.Scatter(x=lr_data.index, y=lr_data.values,
                         name='Learning Rate', line=dict(color='black')),
                row=2, col=1
            )
        
        # 4. Training statistics (box plots)
        if val_metrics:
            val_losses = []
            metric_names = []
            for col in val_metrics:
                data = log_data[col].dropna()
                if len(data) > 0:
                    val_losses.extend(data.values)
                    metric_names.extend([col.replace('val_', '')] * len(data))
            
            if val_losses:
                df_stats = pd.DataFrame({'Metric': metric_names, 'Value': val_losses})
                for metric in df_stats['Metric'].unique():
                    metric_data = df_stats[df_stats['Metric'] == metric]['Value']
                    fig.add_trace(
                        go.Box(y=metric_data, name=metric, boxpoints='outliers'),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive Training Progress Dashboard",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=2)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "training_dashboard.html")
        fig.show()
        
        print(f"Interactive training dashboard saved to: {self.results_dir / 'interactive' / 'training_dashboard.html'}")
    
    def run_training_analysis(self, log_path: str):
        """
        Run complete training analysis
        
        Args:
            log_path: Path to training log file or directory
        """
        print("Starting comprehensive training analysis...")
        print("=" * 60)
        
        # Load training logs
        log_data = self.load_training_logs(log_path)
        print(f"Loaded training logs with {len(log_data)} records")
        print(f"Available columns: {list(log_data.columns)}")
        
        # Run analysis
        self.plot_training_curves(log_data)
        convergence_info = self.analyze_convergence(log_data)
        self.plot_convergence_analysis(log_data, convergence_info)
        self.create_interactive_training_dashboard(log_data)
        
        print("\n" + "=" * 60)
        print(f"Training analysis complete! Results saved to: {self.results_dir}")
        
        return convergence_info


def analyze_training_progress(log_path: str, results_dir: str = "./analysis_results"):
    """
    Convenience function to analyze training progress
    
    Args:
        log_path: Path to training log file or directory
        results_dir: Directory to save results
    """
    diagnostics = TrainingDiagnostics(results_dir)
    return diagnostics.run_training_analysis(log_path)


if __name__ == "__main__":
    # Example usage - you'll need to provide the actual log path
    log_path = "./wandb"  # or specific CSV file path
    analyze_training_progress(log_path)