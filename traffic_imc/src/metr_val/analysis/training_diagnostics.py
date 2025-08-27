"""
Training diagnostics and model performance analysis
Analyzes training curves, convergence patterns, and prediction intervals with Plotly-based interactive visualizations
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


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
        (self.results_dir / "interactive").mkdir(exist_ok=True)
        
        # Plotly configuration
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]  # Use hex colors to avoid conversion issues
        
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
    
    def _hex_to_rgba(self, hex_color: str, alpha: float = 1.0) -> str:
        """
        Convert hex color to rgba string safely
        
        Args:
            hex_color: Hex color string (e.g., '#1f77b4')
            alpha: Alpha value (0.0 to 1.0)
            
        Returns:
            RGBA color string
        """
        try:
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            
            # Convert hex to RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            return f'rgba({r},{g},{b},{alpha})'
        except (ValueError, IndexError):
            # Fallback to a default color
            return f'rgba(31,119,180,{alpha})'  # Default blue
    
    def create_training_curves_dashboard(self, log_data: pd.DataFrame):
        """
        Create interactive training curves dashboard
        
        Args:
            log_data: DataFrame with training metrics
        """
        print("Creating interactive training curves dashboard...")
        
        # Identify available metrics
        metric_columns = [col for col in log_data.columns if any(metric in col.lower() 
                         for metric in ['loss', 'mae', 'rmse', 'mse'])]
        
        # Separate train and validation metrics
        train_metrics = [col for col in metric_columns if 'train' in col.lower()]
        val_metrics = [col for col in metric_columns if 'val' in col.lower()]
        
        if not train_metrics and not val_metrics:
            print("No training metrics found in log data")
            return
            
        # Create subplot structure
        n_metrics = max(len(train_metrics), len(val_metrics))
        subplot_titles = []
        
        for i in range(n_metrics):
            if i < len(train_metrics):
                metric_name = train_metrics[i].replace('train_', '').replace('val_', '').upper()
                subplot_titles.append(f'{metric_name} Progress')
            else:
                subplot_titles.append(f'Metric {i+1} Progress')
        
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1/n_metrics if n_metrics > 1 else 0.1,
            shared_xaxes=True
        )
        
        # Plot each metric pair
        for i in range(n_metrics):
            row = i + 1
            
            # Plot training metric
            if i < len(train_metrics) and train_metrics[i] in log_data.columns:
                train_data = log_data[train_metrics[i]].dropna()
                if len(train_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=train_data.index,
                            y=train_data.values,
                            mode='lines',
                            name=f'Training {train_metrics[i].replace("train_", "")}',
                            line=dict(color=self.color_palette[0], width=2),
                            legendgroup=f"metric_{i}"
                        ),
                        row=row, col=1
                    )
            
            # Plot validation metric
            if i < len(val_metrics) and val_metrics[i] in log_data.columns:
                val_data = log_data[val_metrics[i]].dropna()
                if len(val_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=val_data.index,
                            y=val_data.values,
                            mode='lines',
                            name=f'Validation {val_metrics[i].replace("val_", "")}',
                            line=dict(color=self.color_palette[1], width=2),
                            legendgroup=f"metric_{i}"
                        ),
                        row=row, col=1
                    )
                    
                    # Add smoothed trend line if enough data points
                    if len(val_data) > 10:
                        window = max(1, len(val_data) // 10)
                        smoothed = val_data.rolling(window=window, center=True).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=smoothed.index,
                                y=smoothed.values,
                                mode='lines',
                                name=f'Smoothed {val_metrics[i].replace("val_", "")}',
                                line=dict(color=self.color_palette[2], width=1, dash='dash'),
                                opacity=0.7,
                                legendgroup=f"metric_{i}"
                            ),
                            row=row, col=1
                        )
        
        # Update layout
        fig.update_layout(
            height=300 * n_metrics + 100,
            title_text="Interactive Training Progress Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update x-axis label only for bottom subplot
        fig.update_xaxes(title_text="Epoch/Step", row=n_metrics, col=1)
        
        # Update y-axis labels
        for i in range(n_metrics):
            fig.update_yaxes(title_text="Value", row=i+1, col=1)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "training_curves_dashboard.html")
        fig.show()
        
        print(f"Interactive training curves dashboard saved to: {self.results_dir / 'interactive' / 'training_curves_dashboard.html'}")
        
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
    
    def create_convergence_analysis_dashboard(self, log_data: pd.DataFrame, convergence_info: Dict):
        """
        Create detailed interactive convergence analysis dashboard
        
        Args:
            log_data: DataFrame with training metrics
            convergence_info: Results from analyze_convergence
        """
        print("Creating interactive convergence analysis dashboard...")
        
        # Find loss columns
        val_loss_cols = [col for col in log_data.columns if 'val' in col.lower() and 'loss' in col.lower()]
        train_loss_cols = [col for col in log_data.columns if 'train' in col.lower() and 'loss' in col.lower()]
        lr_cols = [col for col in log_data.columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Progress with Best Epoch', 'Cumulative Loss Improvement',
                          'Validation Loss Change Rate', 'Learning Rate Schedule' if lr_cols else 'Loss Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if val_loss_cols:
            val_loss = log_data[val_loss_cols[0]].dropna()
            
            # 1. Loss curve with best epoch marked
            fig.add_trace(
                go.Scatter(
                    x=val_loss.index,
                    y=val_loss.values,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color=self.color_palette[0], width=2)
                ),
                row=1, col=1
            )
            
            if train_loss_cols:
                train_loss = log_data[train_loss_cols[0]].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=train_loss.index,
                        y=train_loss.values,
                        mode='lines',
                        name='Training Loss',
                        line=dict(color=self.color_palette[1], width=2)
                    ),
                    row=1, col=1
                )
            
            # Mark best epoch
            best_epoch = convergence_info.get('best_epoch', 0)
            best_loss = convergence_info.get('best_val_loss', 0)
            if best_epoch < len(val_loss):
                fig.add_trace(
                    go.Scatter(
                        x=[best_epoch],
                        y=[best_loss],
                        mode='markers',
                        name=f'Best Epoch ({best_epoch})',
                        marker=dict(color='red', size=12, symbol='star')
                    ),
                    row=1, col=1
                )
            
            # 2. Loss improvement over time
            initial_loss = convergence_info.get('initial_val_loss', val_loss.iloc[0] if len(val_loss) > 0 else 0)
            if initial_loss:
                improvement = initial_loss - val_loss
                fig.add_trace(
                    go.Scatter(
                        x=improvement.index,
                        y=improvement.values,
                        mode='lines',
                        name='Loss Improvement',
                        line=dict(color='green', width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # 3. Loss derivatives (learning speed)
            if len(val_loss) > 1:
                loss_diff = val_loss.diff()
                fig.add_trace(
                    go.Scatter(
                        x=loss_diff.index[1:],
                        y=loss_diff.values[1:],
                        mode='lines',
                        name='Loss Change Rate',
                        line=dict(color=self.color_palette[2], width=1),
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Learning rate schedule or loss distribution
        if lr_cols:
            lr_data = log_data[lr_cols[0]].dropna()
            fig.add_trace(
                go.Scatter(
                    x=lr_data.index,
                    y=lr_data.values,
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='orange', width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
            fig.update_yaxes(type="log", row=2, col=2)
        elif val_loss_cols:
            # Plot loss distribution instead
            fig.add_trace(
                go.Histogram(
                    x=val_loss.values,
                    name='Loss Distribution',
                    nbinsx=30,
                    marker_color=self.color_palette[3],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive Model Convergence Analysis",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss Improvement", row=1, col=2)
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss Δ", row=2, col=1)
        
        if lr_cols:
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            fig.update_yaxes(title_text="Learning Rate", row=2, col=2)
        else:
            fig.update_xaxes(title_text="Loss Value", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "convergence_analysis_dashboard.html")
        fig.show()
        
        # Create summary statistics table
        summary_data = [
            ['Best Epoch', str(convergence_info.get('best_epoch', 'N/A'))],
            ['Best Validation Loss', f"{convergence_info.get('best_val_loss', 0):.6f}" if convergence_info.get('best_val_loss') else 'N/A'],
            ['Total Improvement (%)', f"{convergence_info.get('improvement_percentage', 0):.2f}%" if convergence_info.get('improvement_percentage') else 'N/A'],
            ['Early Stopping Detected', str(convergence_info.get('early_stopping_detected', 'N/A'))],
            ['Overfitting Detected', str(convergence_info.get('overfitting_detected', 'N/A'))]
        ]
        
        if convergence_info.get('learning_rate_analysis'):
            lr_info = convergence_info['learning_rate_analysis']
            summary_data.extend([
                ['Learning Rate Reductions', str(lr_info.get('lr_reductions', 'N/A'))],
                ['Final Learning Rate', f"{lr_info.get('final_lr', 0):.2e}" if lr_info.get('final_lr') else 'N/A']
            ])
        
        # Create summary table
        summary_fig = go.Figure(data=[go.Table(
            header=dict(values=['Convergence Metric', 'Value'],
                       fill_color='lightblue',
                       align='left'),
            cells=dict(values=[[item[0] for item in summary_data],
                              [item[1] for item in summary_data]],
                      fill_color='white',
                      align='left'))
        ])
        
        summary_fig.update_layout(
            title="Convergence Analysis Summary",
            width=600,
            height=300
        )
        summary_fig.write_html(self.results_dir / "interactive" / "convergence_summary.html")
        summary_fig.show()
        
        print(f"Interactive convergence analysis dashboard saved to: {self.results_dir / 'interactive' / 'convergence_analysis_dashboard.html'}")
        print(f"Convergence summary saved to: {self.results_dir / 'interactive' / 'convergence_summary.html'}")
        
        # Print convergence summary
        print("\nConvergence Analysis Summary:")
        for metric, value in summary_data:
            print(f"{metric}: {value}")
    
    def create_comprehensive_training_dashboard(self, log_data: pd.DataFrame):
        """
        Create comprehensive interactive training progress dashboard with enhanced features
        
        Args:
            log_data: DataFrame with training metrics
        """
        print("Creating comprehensive interactive training dashboard...")
        
        # Identify metrics
        metric_columns = [col for col in log_data.columns if any(metric in col.lower() 
                         for metric in ['loss', 'mae', 'rmse', 'mse', 'lr', 'learning_rate'])]
        
        train_metrics = [col for col in metric_columns if 'train' in col.lower()]
        val_metrics = [col for col in metric_columns if 'val' in col.lower()]
        lr_metrics = [col for col in metric_columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Metrics Comparison', 'Learning Rate Schedule', 'Training Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Loss curves with enhanced styling
        loss_train_col = None
        loss_val_col = None
        
        for train_col in train_metrics:
            if 'loss' in train_col.lower():
                loss_train_col = train_col
                train_data = log_data[train_col].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=train_data.index, 
                        y=train_data.values,
                        mode='lines',
                        name='Training Loss', 
                        line=dict(color=self.color_palette[0], width=2)
                    ),
                    row=1, col=1
                )
                break
        
        for val_col in val_metrics:
            if 'loss' in val_col.lower():
                loss_val_col = val_col
                val_data = log_data[val_col].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=val_data.index, 
                        y=val_data.values,
                        mode='lines',
                        name='Validation Loss', 
                        line=dict(color=self.color_palette[1], width=2)
                    ),
                    row=1, col=1
                )
                break
        
        # 2. Other metrics comparison with better color management
        other_val_metrics = [col for col in val_metrics if 'loss' not in col.lower()][:4]
        for i, metric_col in enumerate(other_val_metrics):
            metric_data = log_data[metric_col].dropna()
            color_idx = (i + 2) % len(self.color_palette)  # Start from index 2 to avoid loss colors
            fig.add_trace(
                go.Scatter(
                    x=metric_data.index, 
                    y=metric_data.values,
                    mode='lines',
                    name=metric_col.replace('val_', '').upper(), 
                    line=dict(color=self.color_palette[color_idx], width=2)
                ),
                row=1, col=2
            )
        
        # 3. Learning rate with log scale
        if lr_metrics:
            lr_data = log_data[lr_metrics[0]].dropna()
            fig.add_trace(
                go.Scatter(
                    x=lr_data.index, 
                    y=lr_data.values,
                    mode='lines',
                    name='Learning Rate', 
                    line=dict(color='orange', width=3),
                    showlegend=False
                ),
                row=2, col=1
            )
            fig.update_yaxes(type="log", row=2, col=1)
        
        # 4. Enhanced training statistics with violin plots
        if val_metrics:
            # Create violin plots for better distribution visualization
            for i, col in enumerate(val_metrics[:4]):  # Limit to 4 metrics for readability
                data = log_data[col].dropna()
                if len(data) > 0:
                    color_idx = i % len(self.color_palette)
                    fig.add_trace(
                        go.Violin(
                            y=data.values,
                            name=col.replace('val_', ''),
                            box_visible=True,
                            line_color=self.color_palette[color_idx],
                            fillcolor=self._hex_to_rgba(self.color_palette[color_idx], 0.3),
                            opacity=0.6,
                            showlegend=False
                        ),
                        row=2, col=2
                    )
        
        # Update layout with enhanced styling
        fig.update_layout(
            height=800,
            title_text=f"Comprehensive Training Progress Dashboard ({len(log_data)} epochs)",
            showlegend=True,
            hovermode='x unified',
            font=dict(size=12)
        )
        
        # Update axes with better styling
        fig.update_xaxes(title_text="Epoch", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Epoch", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Epoch", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Metrics", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Metric Value", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Learning Rate", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Value Distribution", row=2, col=2)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "comprehensive_training_dashboard.html")
        fig.show()
        
        print(f"Comprehensive training dashboard saved to: {self.results_dir / 'interactive' / 'comprehensive_training_dashboard.html'}")
        print(f"Dashboard shows training progress over {len(log_data)} epochs with {len(metric_columns)} metrics tracked")
    
    def run_complete_training_analysis(self, log_path: str):
        """
        Run complete interactive training analysis with performance optimization
        
        Args:
            log_path: Path to training log file or directory
        """
        print("Starting comprehensive interactive training analysis...")
        print("=" * 80)
        
        # Load training logs
        log_data = self.load_training_logs(log_path)
        print(f"Loaded training logs with {len(log_data)} records")
        print(f"Available columns: {list(log_data.columns)}")
        
        # Identify available metrics
        metric_columns = [col for col in log_data.columns if any(metric in col.lower() 
                         for metric in ['loss', 'mae', 'rmse', 'mse', 'lr', 'learning_rate'])]
        train_metrics = [col for col in metric_columns if 'train' in col.lower()]
        val_metrics = [col for col in metric_columns if 'val' in col.lower()]
        
        print(f"Found {len(train_metrics)} training metrics and {len(val_metrics)} validation metrics")
        
        # Run comprehensive analysis
        print("\n📈 Creating Training Curves Dashboard...")
        self.create_training_curves_dashboard(log_data)
        
        print("\n🎯 Analyzing Model Convergence...")
        convergence_info = self.analyze_convergence(log_data)
        self.create_convergence_analysis_dashboard(log_data, convergence_info)
        
        print("\n🖥️ Creating Comprehensive Training Dashboard...")
        self.create_comprehensive_training_dashboard(log_data)
        
        print("\n" + "=" * 80)
        print("🎉 INTERACTIVE TRAINING ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"📁 Results saved to: {self.results_dir.absolute()}")
        print("\n🌐 Generated Interactive Dashboards:")
        print("   - interactive/training_curves_dashboard.html (Individual metric progress)")
        print("   - interactive/convergence_analysis_dashboard.html (Convergence patterns)")
        print("   - interactive/comprehensive_training_dashboard.html (Complete overview)")
        print("   - interactive/convergence_summary.html (Convergence statistics)")
        print("\n📖 Analysis Features:")
        print(f"   - {len(log_data)} training epochs analyzed")
        print(f"   - {len(train_metrics)} training metrics tracked")
        print(f"   - {len(val_metrics)} validation metrics tracked")
        print(f"   - Early stopping detection: {convergence_info.get('early_stopping_detected', 'N/A')}")
        print(f"   - Overfitting detection: {convergence_info.get('overfitting_detected', 'N/A')}")
        print("\n🚀 How to use:")
        print("   1. Open HTML files in your web browser")
        print("   2. Hover over plots for detailed metrics")
        print("   3. Use zoom and pan for detailed inspection")
        print("   4. Check convergence summary for training insights")
        print("=" * 80)
        
        return convergence_info


def analyze_training_progress(log_path: str, results_dir: str = "./analysis_results"):
    """
    Convenience function to analyze training progress with interactive dashboards
    
    Args:
        log_path: Path to training log file or directory
        results_dir: Directory to save results
    """
    diagnostics = TrainingDiagnostics(results_dir)
    return diagnostics.run_complete_training_analysis(log_path)


if __name__ == "__main__":
    # Example usage - you'll need to provide the actual log path
    log_path = "./wandb"  # or specific CSV file path
    analyze_training_progress(log_path)