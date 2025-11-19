"""
Comprehensive visualization analysis for RNN model predictions
Provides multi-angle analysis of traffic prediction performance with Plotly-based interactive visualizations
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import torch
import lightning as L
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

from metr.datasets.rnn.datamodule import SimpleTrafficDataModule
from ..models.rnn.module import LSTMLightningModule


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
        
        # Plotly configuration
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]  # Use hex colors to avoid conversion issues
        self.chunk_size = 2000  # Optimal chunk size for performance
        
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
    
    def _create_time_chunks(self, data: Dict, chunk_size: int = None) -> List[Dict]:
        """
        Split data into time-based chunks for better performance
        
        Args:
            data: Prediction data dictionary
            chunk_size: Number of samples per chunk
            
        Returns:
            List of data chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        n_samples = len(data['predictions'])
        chunks = []
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            
            chunk = {}
            for key, values in data.items():
                if isinstance(values, np.ndarray):
                    chunk[key] = values[i:end_idx]
                elif hasattr(values, '__getitem__'):  # For pandas Series/Index
                    chunk[key] = values[i:end_idx]
                else:
                    chunk[key] = values
            
            # Add chunk metadata
            chunk['chunk_id'] = len(chunks)
            chunk['start_idx'] = i
            chunk['end_idx'] = end_idx
            chunk['chunk_size'] = end_idx - i
            
            chunks.append(chunk)
        
        return chunks
    
    def _get_adaptive_sample_size(self, total_size: int, max_points: int = 5000) -> int:
        """
        Calculate adaptive sample size based on total data size
        """
        if total_size <= max_points:
            return total_size
        
        # Use logarithmic scaling for very large datasets
        if total_size > 50000:
            return min(max_points, int(np.sqrt(total_size) * 50))
        
        return max_points
    
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
    
    def create_time_series_dashboard(self, max_points_per_chunk: int = 2000):
        """Create interactive time series comparison with chunked data for better performance"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating interactive time series dashboard...")
        
        data = self.predictions
        total_samples = len(data['predictions'])
        
        # Create data chunks for performance optimization
        chunks = self._create_time_chunks(data, self.chunk_size)
        
        # Create main dashboard with dropdown for chunk selection
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Scaled Data: Actual vs Predicted Traffic Volume',
                          'Original Scale: Actual vs Predicted Traffic Volume',
                          'Prediction Residuals Over Time'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Add traces for each chunk (initially hidden except first)
        for i, chunk in enumerate(chunks):
            is_visible = (i == 0)  # Only first chunk visible initially
            
            timestamps = chunk['timestamps']
            
            # Plot 1: Scaled data comparison
            fig.add_trace(
                go.Scatter(
                    x=timestamps, 
                    y=chunk['actuals'],
                    name=f'Actual (Chunk {i+1})',
                    line=dict(color=self.color_palette[0], width=1),
                    visible=is_visible,
                    legendgroup="scaled",
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps, 
                    y=chunk['predictions'],
                    name=f'Predicted (Chunk {i+1})',
                    line=dict(color=self.color_palette[1], width=1),
                    visible=is_visible,
                    legendgroup="scaled",
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )
            
            # Plot 2: Original scale data (if available)
            if 'predictions_original' in chunk:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps, 
                        y=chunk['actuals_original'],
                        name=f'Actual Original (Chunk {i+1})',
                        line=dict(color=self.color_palette[0], width=1),
                        visible=is_visible,
                        legendgroup="original",
                        showlegend=(i == 0)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps, 
                        y=chunk['predictions_original'],
                        name=f'Predicted Original (Chunk {i+1})',
                        line=dict(color=self.color_palette[1], width=1),
                        visible=is_visible,
                        legendgroup="original",
                        showlegend=(i == 0)
                    ),
                    row=2, col=1
                )
            
            # Plot 3: Residuals
            residuals = chunk.get('residuals_original', chunk['residuals'])
            fig.add_trace(
                go.Scatter(
                    x=timestamps, 
                    y=residuals,
                    name=f'Residuals (Chunk {i+1})',
                    line=dict(color=self.color_palette[2], width=1),
                    visible=is_visible,
                    legendgroup="residuals",
                    showlegend=(i == 0)
                ),
                row=3, col=1
            )
        
        # Add zero reference line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)
        
        # Create dropdown menu for chunk selection
        dropdown_buttons = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            start_time = chunk['timestamps'][0].strftime('%Y-%m-%d %H:%M')
            end_time = chunk['timestamps'][-1].strftime('%Y-%m-%d %H:%M')
            
            # Create visibility array
            visibility = [False] * len(fig.data)
            
            # Each chunk has 6 traces (2 for scaled, 2 for original, 1 for residuals, 1 for reference line)
            traces_per_chunk = 5 if 'predictions_original' in chunks[0] else 3
            start_idx = i * traces_per_chunk
            end_idx = start_idx + traces_per_chunk
            
            for j in range(start_idx, min(end_idx, len(visibility))):
                visibility[j] = True
            
            dropdown_buttons.append(
                dict(
                    label=f"Time Period {i+1}: {start_time} - {end_time}",
                    method="update",
                    args=[{"visible": visibility}]
                )
            )
        
        # Add "Show All" option with adaptive sampling
        if len(chunks) > 1:
            sample_size = self._get_adaptive_sample_size(total_samples)
            if sample_size < total_samples:
                indices = np.linspace(0, total_samples-1, sample_size, dtype=int)
                
                # Add sampled full data traces
                timestamps_full = data['timestamps'][indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps_full, 
                        y=data['actuals'][indices],
                        name='Actual (Sampled)',
                        line=dict(color=self.color_palette[0], width=1),
                        visible=False,
                        legendgroup="full"
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps_full, 
                        y=data['predictions'][indices],
                        name='Predicted (Sampled)',
                        line=dict(color=self.color_palette[1], width=1),
                        visible=False,
                        legendgroup="full"
                    ),
                    row=1, col=1
                )
                
                if 'predictions_original' in data:
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps_full, 
                            y=data['actuals_original'][indices],
                            name='Actual Original (Sampled)',
                            line=dict(color=self.color_palette[0], width=1),
                            visible=False,
                            legendgroup="full"
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps_full, 
                            y=data['predictions_original'][indices],
                            name='Predicted Original (Sampled)',
                            line=dict(color=self.color_palette[1], width=1),
                            visible=False,
                            legendgroup="full"
                        ),
                        row=2, col=1
                    )
                
                residuals_full = data.get('residuals_original', data['residuals'])[indices]
                fig.add_trace(
                    go.Scatter(
                        x=timestamps_full, 
                        y=residuals_full,
                        name='Residuals (Sampled)',
                        line=dict(color=self.color_palette[2], width=1),
                        visible=False,
                        legendgroup="full"
                    ),
                    row=3, col=1
                )
                
                # Add "Show All (Sampled)" option
                full_visibility = [False] * len(fig.data)
                # Show only the last few traces (full data traces)
                traces_per_chunk = 5 if 'predictions_original' in data else 3
                for j in range(len(full_visibility) - traces_per_chunk, len(full_visibility)):
                    full_visibility[j] = True
                
                dropdown_buttons.append(
                    dict(
                        label=f"Full Dataset (Sampled - {sample_size:,} points)",
                        method="update",
                        args=[{"visible": full_visibility}]
                    )
                )
        
        # Update layout with dropdown
        fig.update_layout(
            height=900,
            title=f"Interactive Time Series Analysis - {total_samples:,} Total Points",
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ],
            annotations=[
                dict(text="Select Time Period:", showarrow=False, 
                     x=0, y=1.08, yref="paper", align="left")
            ]
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Scaled Traffic Volume", row=1, col=1)
        fig.update_yaxes(title_text="Traffic Volume", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=3, col=1)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "time_series_dashboard.html")
        fig.show()
        
        print(f"Interactive time series dashboard saved to: {self.results_dir / 'interactive' / 'time_series_dashboard.html'}")
        print(f"Dashboard contains {len(chunks)} time periods with {self.chunk_size:,} points each")
        
    def create_error_analysis_dashboard(self):
        """Create comprehensive interactive error analysis dashboard"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating interactive error analysis dashboard...")
        
        data = self.predictions
        
        # Use original scale data if available, otherwise scaled
        residuals = data.get('residuals_original', data['residuals'])
        abs_errors = data.get('absolute_errors_original', data['absolute_errors'])
        actuals = data.get('actuals_original', data['actuals'])
        predictions = data.get('predictions_original', data['predictions'])
        timestamps = data['timestamps']
        
        # Adaptive sampling for scatter plots to improve performance
        total_samples = len(residuals)
        sample_size = self._get_adaptive_sample_size(total_samples, max_points=10000)
        
        if sample_size < total_samples:
            sample_indices = np.random.choice(total_samples, sample_size, replace=False)
            sample_indices = np.sort(sample_indices)
        else:
            sample_indices = np.arange(total_samples)
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Residuals Distribution', 'Q-Q Plot: Residuals Normality', 
                          'Residuals vs Fitted Values', 'Absolute Errors Over Time',
                          'Error Distribution by Traffic Volume', 'Actual vs Predicted Scatter'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Residuals histogram
        fig.add_trace(
            go.Histogram(
                x=residuals, 
                nbinsx=50, 
                name='Residuals',
                marker_color=self.color_palette[0],
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Q-Q plot for residuals normality
        qq_result = stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq_result[0][0]
        sample_quantiles = qq_result[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(size=3, color=self.color_palette[1])
            ),
            row=1, col=2
        )
        
        # Add reference line for Q-Q plot
        qq_line = stats.linregress(theoretical_quantiles, sample_quantiles)
        ref_line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        ref_line_y = qq_line.slope * ref_line_x + qq_line.intercept
        fig.add_trace(
            go.Scatter(
                x=ref_line_x,
                y=ref_line_y,
                mode='lines',
                name='Reference Line',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Residuals vs Fitted values (sampled)
        fig.add_trace(
            go.Scatter(
                x=predictions[sample_indices],
                y=residuals[sample_indices],
                mode='markers',
                name='Residuals vs Fitted',
                marker=dict(size=3, color=self.color_palette[2], opacity=0.6)
            ),
            row=1, col=3
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=3)
        
        # 4. Absolute errors over time (chunked for performance)
        if len(abs_errors) > 10000:
            # Use chunked approach for time series
            chunk_size = max(1, len(abs_errors) // 5000)  # Downsample to ~5000 points
            chunked_errors = abs_errors[::chunk_size]
            chunked_timestamps = timestamps[::chunk_size]
        else:
            chunked_errors = abs_errors
            chunked_timestamps = timestamps
            
        fig.add_trace(
            go.Scatter(
                x=chunked_timestamps,
                y=chunked_errors,
                mode='lines',
                name='Absolute Errors',
                line=dict(color=self.color_palette[3], width=1)
            ),
            row=2, col=1
        )
        
        # 5. Error distribution by traffic volume ranges (box plot)
        quantiles = np.quantile(actuals, [0, 0.25, 0.5, 0.75, 1.0])
        range_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
        
        # Create box plot data
        box_data = []
        for i in range(len(quantiles)-1):
            mask = (actuals >= quantiles[i]) & (actuals < quantiles[i+1])
            if i == len(quantiles)-2:  # Include the maximum value in the last bin
                mask = (actuals >= quantiles[i]) & (actuals <= quantiles[i+1])
            
            range_errors = abs_errors[mask]
            if len(range_errors) > 0:
                fig.add_trace(
                    go.Box(
                        y=range_errors,
                        name=range_labels[i],
                        boxpoints='outliers',
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=2, col=2
                )
        
        # 6. Prediction accuracy scatter plot (sampled)
        fig.add_trace(
            go.Scatter(
                x=actuals[sample_indices],
                y=predictions[sample_indices],
                mode='markers',
                name='Predictions',
                marker=dict(size=3, color=self.color_palette[0], opacity=0.6)
            ),
            row=2, col=3
        )
        
        # Add perfect prediction line
        min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Interactive Error Analysis Dashboard - {total_samples:,} Data Points",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Residuals", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=3)
        fig.update_yaxes(title_text="Residuals", row=1, col=3)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Absolute Error", row=2, col=1)
        
        fig.update_xaxes(title_text="Traffic Volume Range", row=2, col=2)
        fig.update_yaxes(title_text="Absolute Error", row=2, col=2)
        
        fig.update_xaxes(title_text="Actual Values", row=2, col=3)
        fig.update_yaxes(title_text="Predicted Values", row=2, col=3)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "error_analysis_dashboard.html")
        fig.show()
        
        # Calculate and display summary statistics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(data.get('squared_errors', data['squared_errors'])))
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        max_abs_error = np.max(abs_errors)
        p95_abs_error = np.percentile(abs_errors, 95)
        
        # Create summary statistics table
        summary_fig = go.Figure(data=[go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='lightblue',
                       align='left'),
            cells=dict(values=[
                ['Mean Absolute Error', 'Root Mean Square Error', 'Mean Residual', 
                 'Std Residual', 'Max Absolute Error', '95th Percentile Absolute Error',
                 'Total Data Points', 'Sampled Points (for scatter plots)'],
                [f'{mae:.4f}', f'{rmse:.4f}', f'{mean_residual:.4f}', 
                 f'{std_residual:.4f}', f'{max_abs_error:.4f}', f'{p95_abs_error:.4f}',
                 f'{total_samples:,}', f'{sample_size:,}']
            ],
            fill_color='white',
            align='left'))
        ])
        
        summary_fig.update_layout(
            title="Error Analysis Summary Statistics",
            width=600,
            height=300
        )
        summary_fig.write_html(self.results_dir / "interactive" / "error_summary.html")
        summary_fig.show()
        
        print(f"Interactive error analysis dashboard saved to: {self.results_dir / 'interactive' / 'error_analysis_dashboard.html'}")
        print(f"Summary statistics saved to: {self.results_dir / 'interactive' / 'error_summary.html'}")
        
        print("\nError Analysis Summary:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Square Error: {rmse:.4f}")
        print(f"Mean Residual: {mean_residual:.4f}")
        print(f"Std Residual: {std_residual:.4f}")
        print(f"Max Absolute Error: {max_abs_error:.4f}")
        print(f"95th Percentile Absolute Error: {p95_abs_error:.4f}")
        print(f"Data points used: {total_samples:,} (sampled to {sample_size:,} for scatter plots)")
    
    def create_temporal_analysis_dashboard(self):
        """Create interactive temporal pattern analysis dashboard"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating interactive temporal pattern analysis dashboard...")
        
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
        
        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Error by Hour of Day', 'Prediction Error by Day of Week',
                          'Weekday vs Weekend Comparison', 'Error Heatmap: Hour vs Day of Week'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "heatmap"}]]
        )
        
        # 1. Error by hour of day
        hourly_errors = df.groupby('hour')['absolute_error'].agg(['mean', 'std']).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=hourly_errors['hour'],
                y=hourly_errors['mean'],
                mode='lines+markers',
                name='Mean Error',
                line=dict(color=self.color_palette[0], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add error band
        upper_bound = hourly_errors['mean'] + hourly_errors['std']
        lower_bound = hourly_errors['mean'] - hourly_errors['std']
        
        fig.add_trace(
            go.Scatter(
                x=list(hourly_errors['hour']) + list(hourly_errors['hour'][::-1]),
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill='toself',
                fillcolor=self._hex_to_rgba(self.color_palette[0], 0.3),
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Error by day of week
        daily_errors = df.groupby(['day_of_week', 'day_name'])['absolute_error'].mean().reset_index()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_errors = daily_errors.set_index('day_name').reindex(day_order).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=daily_errors['day_name'],
                y=daily_errors['absolute_error'],
                name='Daily Error',
                marker_color=self.color_palette[1]
            ),
            row=1, col=2
        )
        
        # 3. Weekend vs Weekday comparison
        weekend_comparison = df.groupby('is_weekend')['absolute_error'].agg(['mean', 'std']).reset_index()
        weekend_comparison['label'] = weekend_comparison['is_weekend'].map({False: 'Weekday', True: 'Weekend'})
        
        fig.add_trace(
            go.Bar(
                x=weekend_comparison['label'],
                y=weekend_comparison['mean'],
                error_y=dict(type='data', array=weekend_comparison['std']),
                name='Weekend vs Weekday',
                marker_color=self.color_palette[2]
            ),
            row=2, col=1
        )
        
        # 4. Heatmap: Hour vs Day of Week
        pivot_table = df.pivot_table(values='absolute_error', index='hour', columns='day_name', aggfunc='mean')
        pivot_table = pivot_table.reindex(columns=day_order)
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=day_order,
                y=list(range(24)),
                colorscale='YlOrRd',
                name='Error Heatmap',
                colorbar=dict(title='Mean Absolute Error')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive Temporal Pattern Analysis of Prediction Errors",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=1)
        
        fig.update_xaxes(title_text="Day of Week", row=1, col=2)
        fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=2)
        
        fig.update_xaxes(title_text="Period Type", row=2, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=2, col=1)
        
        fig.update_xaxes(title_text="Day of Week", row=2, col=2)
        fig.update_yaxes(title_text="Hour of Day", row=2, col=2)
        
        # Save and show
        fig.write_html(self.results_dir / "interactive" / "temporal_analysis_dashboard.html")
        fig.show()
        
        # Create summary statistics
        summary_stats = {
            'highest_error_hour': hourly_errors.loc[hourly_errors['mean'].idxmax(), 'hour'],
            'highest_error_value': hourly_errors['mean'].max(),
            'lowest_error_hour': hourly_errors.loc[hourly_errors['mean'].idxmin(), 'hour'],
            'lowest_error_value': hourly_errors['mean'].min(),
            'weekend_avg_error': weekend_comparison[weekend_comparison['is_weekend']]['mean'].iloc[0],
            'weekday_avg_error': weekend_comparison[~weekend_comparison['is_weekend']]['mean'].iloc[0]
        }
        
        # Create summary table
        summary_fig = go.Figure(data=[go.Table(
            header=dict(values=['Temporal Pattern', 'Result'],
                       fill_color='lightblue',
                       align='left'),
            cells=dict(values=[
                ['Highest Error Hour', 'Highest Error Value', 'Lowest Error Hour', 
                 'Lowest Error Value', 'Weekend Average Error', 'Weekday Average Error'],
                [f"{summary_stats['highest_error_hour']:02d}:00",
                 f"{summary_stats['highest_error_value']:.4f}",
                 f"{summary_stats['lowest_error_hour']:02d}:00",
                 f"{summary_stats['lowest_error_value']:.4f}",
                 f"{summary_stats['weekend_avg_error']:.4f}",
                 f"{summary_stats['weekday_avg_error']:.4f}"]
            ],
            fill_color='white',
            align='left'))
        ])
        
        summary_fig.update_layout(
            title="Temporal Analysis Summary",
            width=600,
            height=250
        )
        summary_fig.write_html(self.results_dir / "interactive" / "temporal_summary.html")
        summary_fig.show()
        
        print(f"Interactive temporal analysis dashboard saved to: {self.results_dir / 'interactive' / 'temporal_analysis_dashboard.html'}")
        print(f"Temporal summary saved to: {self.results_dir / 'interactive' / 'temporal_summary.html'}")
        
        # Print temporal insights
        print("\nTemporal Analysis Summary:")
        print(f"Highest error hour: {summary_stats['highest_error_hour']:02d}:00 "
              f"(MAE: {summary_stats['highest_error_value']:.4f})")
        print(f"Lowest error hour: {summary_stats['lowest_error_hour']:02d}:00 "
              f"(MAE: {summary_stats['lowest_error_value']:.4f})")
        print(f"Weekend average error: {summary_stats['weekend_avg_error']:.4f}")
        print(f"Weekday average error: {summary_stats['weekday_avg_error']:.4f}")
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive interactive dashboard with all analyses"""
        if self.predictions is None:
            self.generate_predictions()
            
        print("Creating comprehensive interactive dashboard...")
        
        data = self.predictions
        total_samples = len(data['predictions'])
        
        # Adaptive sampling for overview
        sample_size = self._get_adaptive_sample_size(total_samples, max_points=3000)
        indices = np.linspace(0, total_samples-1, sample_size, dtype=int)
        
        # Prepare data
        timestamps = data['timestamps'][indices]
        actuals = data.get('actuals_original', data['actuals'])[indices]
        predictions = data.get('predictions_original', data['predictions'])[indices]
        residuals = data.get('residuals_original', data['residuals'])[indices]
        abs_errors = data.get('absolute_errors_original', data['absolute_errors'])
        
        # Create main dashboard with multiple tabs via dropdown
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series Overview', 'Residuals Over Time', 
                          'Error Distribution', 'Actual vs Predicted'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Time series comparison
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=actuals, 
                name='Actual', 
                line=dict(color=self.color_palette[0], width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=predictions, 
                name='Predicted', 
                line=dict(color=self.color_palette[1], width=2)
            ),
            row=1, col=1
        )
        
        # 2. Residuals over time
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=residuals, 
                name='Residuals', 
                line=dict(color=self.color_palette[2], width=1)
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
        
        # 3. Error distribution
        fig.add_trace(
            go.Histogram(
                x=abs_errors, 
                name='Error Distribution', 
                nbinsx=50,
                marker_color=self.color_palette[3],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 4. Actual vs Predicted scatter
        fig.add_trace(
            go.Scatter(
                x=actuals, y=predictions, 
                mode='markers', name='Predictions',
                marker=dict(size=4, opacity=0.6, color=self.color_palette[0])
            ),
            row=2, col=2
        )
        
        # Add perfect prediction line
        min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout with enhanced interactivity
        fig.update_layout(
            height=800,
            title_text=f"Comprehensive RNN Analysis Dashboard ({total_samples:,} total points, {sample_size:,} shown)",
            showlegend=True,
            hovermode='x unified'
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
        fig.write_html(self.results_dir / "interactive" / "comprehensive_dashboard.html")
        fig.show()
        
        print(f"Comprehensive dashboard saved to: {self.results_dir / 'interactive' / 'comprehensive_dashboard.html'}")
        print(f"Dashboard shows overview of {sample_size:,} sampled points from {total_samples:,} total points")
        print("For detailed analysis, use the individual dashboards (time_series, error_analysis, temporal_analysis)")
    
    def run_complete_analysis(self):
        """Run all interactive analysis methods with performance optimization"""
        print("Starting comprehensive RNN prediction analysis with interactive dashboards...")
        print("=" * 80)
        
        # Load model and generate predictions
        self.load_model_and_data()
        self.generate_predictions()
        
        total_points = len(self.predictions['predictions'])
        print(f"Generated {total_points:,} prediction points for analysis")
        print(f"Using chunk size of {self.chunk_size:,} for optimized performance")
        
        # Run all analysis with performance optimization
        print("\n📊 Creating Time Series Dashboard...")
        self.create_time_series_dashboard()
        
        print("\n📈 Creating Error Analysis Dashboard...")
        self.create_error_analysis_dashboard()
        
        print("\n⏰ Creating Temporal Analysis Dashboard...")
        self.create_temporal_analysis_dashboard()
        
        print("\n🎯 Creating Comprehensive Overview Dashboard...")
        self.create_comprehensive_dashboard()
        
        print("\n" + "=" * 80)
        print("🎉 COMPLETE INTERACTIVE ANALYSIS FINISHED!")
        print("=" * 80)
        print(f"📁 Results saved to: {self.results_dir.absolute()}")
        print("\n🌐 Generated Interactive Dashboards:")
        print("   - interactive/time_series_dashboard.html (Chunked time series with period selection)")
        print("   - interactive/error_analysis_dashboard.html (Comprehensive error analysis)")
        print("   - interactive/temporal_analysis_dashboard.html (Time pattern analysis)")
        print("   - interactive/comprehensive_dashboard.html (Overview dashboard)")
        print("   - interactive/error_summary.html (Error statistics table)")
        print("   - interactive/temporal_summary.html (Temporal statistics table)")
        print("\n💾 Data files:")
        print("   - data/predictions.pkl (Complete prediction data)")
        print("\n📖 Performance Optimizations Applied:")
        print(f"   - Time series chunked into {len(self._create_time_chunks(self.predictions))} periods")
        print(f"   - Adaptive sampling for scatter plots (up to {self._get_adaptive_sample_size(total_points):,} points)")
        print(f"   - Optimized rendering for {total_points:,} total data points")
        print("\n🚀 How to use:")
        print("   1. Open HTML files in your web browser for interactive exploration")
        print("   2. Use dropdown menus to navigate different time periods")
        print("   3. Hover over plots for detailed information")
        print("   4. Use zoom and pan for detailed analysis")
        print("=" * 80)


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