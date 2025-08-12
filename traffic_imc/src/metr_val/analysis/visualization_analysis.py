"""
Comprehensive visualization analysis for RNN model predictions
Provides multi-angle analysis of traffic prediction performance
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import lightning as L
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

from metr.datasets.rnn.datamodule import SimpleTrafficDataModule
from ..models.rnn import LSTMLightningModule


class RNNPredictionAnalyzer:
    """Comprehensive analysis tool for RNN traffic prediction models"""
    
    def __init__(self, model_checkpoint_path: str, data_path: str, results_dir: str = "./analysis_results"):
        """
        Initialize the analyzer
        
        Args:
            model_checkpoint_path: Path to trained model checkpoint
            data_path: Path to data file (HDF5)
            results_dir: Directory to save analysis results
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of analysis
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "interactive").mkdir(exist_ok=True)
        (self.results_dir / "data").mkdir(exist_ok=True)
        
        # Model and data will be loaded on demand
        self.model = None
        self.data_module = None
        self.predictions = None
        self.metadata = None
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_model_and_data(self):
        """Load trained model and prepare data"""
        print("Loading model and data...")
        
        # Load data module
        self.data_module = SimpleTrafficDataModule(self.data_path)
        self.data_module.setup('test')
        
        # Load trained model
        self.model = LSTMLightningModule.load_from_checkpoint(
            self.model_checkpoint_path,
            scaler=self.data_module.scaler
        )
        self.model.eval()
        
        # Move model to available device
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        print(f"Model moved to device: {device}")
        
        print(f"Model loaded from: {self.model_checkpoint_path}")
        print(f"Data loaded from: {self.data_path}")
        
    def generate_predictions(self) -> Dict[str, np.ndarray]:
        """Generate predictions on test set"""
        if self.model is None:
            self.load_model_and_data()
            
        print("Generating predictions...")
        
        test_loader = self.data_module.test_dataloader()
        predictions = []
        actuals = []
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                # Move input data to same device as model
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                
                # Convert to numpy and flatten
                pred_np = pred.cpu().numpy()
                actual_np = y.cpu().numpy()
                
                if actual_np.ndim > 2:
                    actual_np = actual_np.squeeze(-1)
                if pred_np.ndim > 2:
                    pred_np = pred_np.squeeze(-1)
                    
                predictions.append(pred_np)
                actuals.append(actual_np)
        
        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Create timestamp index (assuming daily data with some frequency)
        n_samples = len(predictions)
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        # Store results
        self.predictions = {
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'residuals': actuals - predictions,
            'absolute_errors': np.abs(actuals - predictions),
            'squared_errors': (actuals - predictions) ** 2
        }
        
        # Calculate inverse transformed values if scaler available
        if self.data_module.scaler is not None:
            pred_orig = self.data_module.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actual_orig = self.data_module.scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
            
            self.predictions.update({
                'predictions_original': pred_orig,
                'actuals_original': actual_orig,
                'residuals_original': actual_orig - pred_orig,
                'absolute_errors_original': np.abs(actual_orig - pred_orig)
            })
        
        # Save predictions data
        with open(self.results_dir / "data" / "predictions.pkl", 'wb') as f:
            pickle.dump(self.predictions, f)
            
        print(f"Generated {n_samples} predictions")
        return self.predictions
    
    def plot_time_series_comparison(self, n_samples: int = 500, sensor_id: Optional[int] = None):
        """Plot actual vs predicted time series"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating time series comparison plots...")
        
        data = self.predictions
        
        # Select subset of data for visibility
        if n_samples < len(data['predictions']):
            indices = np.random.choice(len(data['predictions']), n_samples, replace=False)
            indices = np.sort(indices)
        else:
            indices = np.arange(len(data['predictions']))
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('RNN Traffic Prediction Analysis: Time Series Comparison', fontsize=16)
        
        timestamps = data['timestamps'][indices]
        
        # Plot 1: Scaled data comparison
        axes[0].plot(timestamps, data['actuals'][indices], label='Actual', alpha=0.8, linewidth=1)
        axes[0].plot(timestamps, data['predictions'][indices], label='Predicted', alpha=0.8, linewidth=1)
        axes[0].set_title('Scaled Data: Actual vs Predicted Traffic Volume')
        axes[0].set_ylabel('Scaled Traffic Volume')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Original scale data (if available)
        if 'predictions_original' in data:
            axes[1].plot(timestamps, data['actuals_original'][indices], label='Actual', alpha=0.8, linewidth=1)
            axes[1].plot(timestamps, data['predictions_original'][indices], label='Predicted', alpha=0.8, linewidth=1)
            axes[1].set_title('Original Scale: Actual vs Predicted Traffic Volume')
            axes[1].set_ylabel('Traffic Volume')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals over time
        residuals = data['residuals_original'][indices] if 'residuals_original' in data else data['residuals'][indices]
        axes[2].plot(timestamps, residuals, color='red', alpha=0.6, linewidth=1)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('Prediction Residuals Over Time')
        axes[2].set_ylabel('Residuals')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "time_series_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_error_analysis(self):
        """Create comprehensive error analysis plots"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating error analysis plots...")
        
        data = self.predictions
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RNN Model Error Analysis', fontsize=16)
        
        # Use original scale data if available, otherwise scaled
        residuals = data.get('residuals_original', data['residuals'])
        abs_errors = data.get('absolute_errors_original', data['absolute_errors'])
        actuals = data.get('actuals_original', data['actuals'])
        predictions = data.get('predictions_original', data['predictions'])
        
        # 1. Residuals histogram
        axes[0, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals Distribution')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot for residuals normality
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot: Residuals Normality')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals vs Fitted values
        axes[0, 2].scatter(predictions, residuals, alpha=0.5, s=1)
        axes[0, 2].axhline(y=0, color='red', linestyle='--')
        axes[0, 2].set_title('Residuals vs Fitted Values')
        axes[0, 2].set_xlabel('Predicted Values')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Absolute errors over time
        time_hours = np.arange(len(abs_errors)) 
        axes[1, 0].plot(time_hours, abs_errors, alpha=0.6, linewidth=1)
        axes[1, 0].set_title('Absolute Errors Over Time')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error distribution by actual value ranges
        # Bin actual values into quantiles
        quantiles = np.quantile(actuals, [0, 0.25, 0.5, 0.75, 1.0])
        bins = pd.cut(actuals, bins=quantiles, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        error_by_range = pd.DataFrame({'Range': bins, 'AbsError': abs_errors})
        
        sns.boxplot(data=error_by_range, x='Range', y='AbsError', ax=axes[1, 1])
        axes[1, 1].set_title('Error Distribution by Traffic Volume Ranges')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Prediction accuracy scatter plot
        axes[1, 2].scatter(actuals, predictions, alpha=0.5, s=1)
        min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        axes[1, 2].set_title('Actual vs Predicted Scatter Plot')
        axes[1, 2].set_xlabel('Actual Values')
        axes[1, 2].set_ylabel('Predicted Values')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "error_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\nError Analysis Summary:")
        print(f"Mean Absolute Error: {np.mean(abs_errors):.4f}")
        print(f"Root Mean Square Error: {np.sqrt(np.mean(data.get('squared_errors', data['squared_errors']))):.4f}")
        print(f"Mean Residual: {np.mean(residuals):.4f}")
        print(f"Std Residual: {np.std(residuals):.4f}")
        print(f"Max Absolute Error: {np.max(abs_errors):.4f}")
        print(f"95th Percentile Absolute Error: {np.percentile(abs_errors, 95):.4f}")
    
    def plot_temporal_analysis(self):
        """Analyze performance by time patterns (hourly, daily, weekly)"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating temporal pattern analysis...")
        
        data = self.predictions
        timestamps = data['timestamps']
        abs_errors = data.get('absolute_errors_original', data['absolute_errors'])
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'timestamp': timestamps,
            'absolute_error': abs_errors,
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'day_name': timestamps.day_name(),
            'is_weekend': timestamps.dayofweek >= 5
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Pattern Analysis of Prediction Errors', fontsize=16)
        
        # 1. Error by hour of day
        hourly_errors = df.groupby('hour')['absolute_error'].agg(['mean', 'std']).reset_index()
        axes[0, 0].plot(hourly_errors['hour'], hourly_errors['mean'], marker='o')
        axes[0, 0].fill_between(hourly_errors['hour'], 
                               hourly_errors['mean'] - hourly_errors['std'],
                               hourly_errors['mean'] + hourly_errors['std'], 
                               alpha=0.3)
        axes[0, 0].set_title('Prediction Error by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # 2. Error by day of week
        daily_errors = df.groupby(['day_of_week', 'day_name'])['absolute_error'].mean().reset_index()
        axes[0, 1].bar(daily_errors['day_name'], daily_errors['absolute_error'])
        axes[0, 1].set_title('Prediction Error by Day of Week')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Weekend vs Weekday comparison
        weekend_comparison = df.groupby('is_weekend')['absolute_error'].agg(['mean', 'std']).reset_index()
        weekend_comparison['label'] = weekend_comparison['is_weekend'].map({False: 'Weekday', True: 'Weekend'})
        
        axes[1, 0].bar(weekend_comparison['label'], weekend_comparison['mean'], 
                      yerr=weekend_comparison['std'], capsize=5)
        axes[1, 0].set_title('Weekday vs Weekend Prediction Error')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Heatmap: Hour vs Day of Week
        pivot_table = df.pivot_table(values='absolute_error', index='hour', columns='day_name', aggfunc='mean')
        # Reorder columns to start with Monday
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(columns=day_order)
        
        sns.heatmap(pivot_table, ax=axes[1, 1], cmap='YlOrRd', cbar_kws={'label': 'Mean Absolute Error'})
        axes[1, 1].set_title('Error Heatmap: Hour vs Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Hour of Day')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "plots" / "temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print temporal insights
        print("\nTemporal Analysis Summary:")
        print(f"Highest error hour: {hourly_errors.loc[hourly_errors['mean'].idxmax(), 'hour']:02d}:00 "
              f"(MAE: {hourly_errors['mean'].max():.4f})")
        print(f"Lowest error hour: {hourly_errors.loc[hourly_errors['mean'].idxmin(), 'hour']:02d}:00 "
              f"(MAE: {hourly_errors['mean'].min():.4f})")
        print(f"Weekend average error: {weekend_comparison[weekend_comparison['is_weekend']]['mean'].iloc[0]:.4f}")
        print(f"Weekday average error: {weekend_comparison[~weekend_comparison['is_weekend']]['mean'].iloc[0]:.4f}")
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating interactive dashboard...")
        
        data = self.predictions
        
        # Prepare data
        timestamps = data['timestamps']
        actuals = data.get('actuals_original', data['actuals'])
        predictions = data.get('predictions_original', data['predictions'])
        residuals = data.get('residuals_original', data['residuals'])
        abs_errors = data.get('absolute_errors_original', data['absolute_errors'])
        
        # Sample data for better performance in interactive plots
        n_samples = min(1000, len(actuals))
        indices = np.random.choice(len(actuals), n_samples, replace=False)
        indices = np.sort(indices)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series Comparison', 'Residuals Over Time', 
                          'Error Distribution', 'Actual vs Predicted'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Time series comparison
        fig.add_trace(
            go.Scatter(x=timestamps[indices], y=actuals[indices], 
                      name='Actual', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=timestamps[indices], y=predictions[indices], 
                      name='Predicted', line=dict(color='red', width=1)),
            row=1, col=1
        )
        
        # 2. Residuals over time
        fig.add_trace(
            go.Scatter(x=timestamps[indices], y=residuals[indices], 
                      name='Residuals', line=dict(color='green', width=1)),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
        
        # 3. Error distribution
        fig.add_trace(
            go.Histogram(x=abs_errors, name='Error Distribution', nbinsx=50),
            row=2, col=1
        )
        
        # 4. Actual vs Predicted scatter
        fig.add_trace(
            go.Scatter(x=actuals[indices], y=predictions[indices], 
                      mode='markers', name='Predictions',
                      marker=dict(size=3, opacity=0.6)),
            row=2, col=2
        )
        
        # Add perfect prediction line
        min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive RNN Prediction Analysis Dashboard",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Absolute Error", row=2, col=1)
        fig.update_xaxes(title_text="Actual Values", row=2, col=2)
        
        fig.update_yaxes(title_text="Traffic Volume", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=2, col=2)
        
        # Save interactive plot
        fig.write_html(self.results_dir / "interactive" / "dashboard.html")
        
        # Show plot
        fig.show()
        
        print(f"Interactive dashboard saved to: {self.results_dir / 'interactive' / 'dashboard.html'}")
    
    def run_complete_analysis(self, n_samples_ts: int = 500):
        """Run all analysis methods"""
        print("Starting comprehensive RNN prediction analysis...")
        print("=" * 60)
        
        # Load model and generate predictions
        self.load_model_and_data()
        self.generate_predictions()
        
        # Run all analysis
        self.plot_time_series_comparison(n_samples=n_samples_ts)
        self.plot_error_analysis()
        self.plot_temporal_analysis()
        self.create_interactive_dashboard()
        
        print("\n" + "=" * 60)
        print(f"Complete analysis finished! Results saved to: {self.results_dir}")
        print("Generated files:")
        print("- plots/time_series_comparison.png")
        print("- plots/error_analysis.png") 
        print("- plots/temporal_analysis.png")
        print("- interactive/dashboard.html")
        print("- data/predictions.pkl")


def analyze_rnn_predictions(model_checkpoint_path: str, 
                          data_path: str, 
                          results_dir: str = "./analysis_results"):
    """
    Convenience function to run complete analysis
    
    Args:
        model_checkpoint_path: Path to trained RNN model checkpoint
        data_path: Path to HDF5 data file
        results_dir: Directory to save analysis results
    """
    analyzer = RNNPredictionAnalyzer(model_checkpoint_path, data_path, results_dir)
    analyzer.run_complete_analysis()
    return analyzer


if __name__ == "__main__":
    # Example usage
    model_path = "./output/rnn/best-epoch=14-val_loss=0.00.ckpt"
    data_path = "./data/selected_small_v1/metr-imc.h5"
    
    analyzer = analyze_rnn_predictions(model_path, data_path)