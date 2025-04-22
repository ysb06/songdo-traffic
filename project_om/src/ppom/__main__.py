import argparse
import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from tqdm import tqdm

from metr.components import Metadata, TrafficData
from metr.components.metr_imc.interpolation import (
    Interpolator,
    LinearInterpolator,
    MonthlyMeanFillInterpolator,
    ShiftFillInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
)
from metr.components.metr_imc.outlier import (
    HourlyInSensorZscoreOutlierProcessor,
    InSensorZscoreOutlierProcessor,
    MADOutlierProcessor,
    OutlierProcessor,
    TrimmedMeanOutlierProcessor,
    WinsorizedOutlierProcessor,
)
# from songdo_rnn.preprocessing.missing import interpolate
# from songdo_rnn.preprocessing.outlier import remove_outliers
from ppom import RAW_DATA_PATH


logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


raw_data = TrafficData.import_from_hdf(RAW_DATA_PATH)
raw_df = raw_data.data


# ------------------ #


class OutlierAndInterpolationEvaluator:
    def __init__(
        self,
        raw_data_path: str,
        output_dir: str = "./output",
        test_start_date: str = "2024-10-01",
        test_end_date: str = "2024-10-31",
    ):
        """
        Initialize the evaluator with data paths and parameters
        """
        # Load data
        self.raw_data = TrafficData.import_from_hdf(raw_data_path)
        self.raw_df = self.raw_data.data

        # Initialize processors
        self._init_processors()

        # Initialize metrics storage
        self.metrics_results = {}
        self.outlier_metrics = {}

    def _init_processors(self):
        """Initialize outlier processors and interpolators"""
        # Outlier processors
        self.outlier_processors = [
            HourlyInSensorZscoreOutlierProcessor(),
            InSensorZscoreOutlierProcessor(),
            WinsorizedOutlierProcessor(),
            TrimmedMeanOutlierProcessor(),
            MADOutlierProcessor(),
        ]

        # Set custom names for clarity
        self.outlier_processors[0].name = "hzscore"
        self.outlier_processors[1].name = "zscore"
        self.outlier_processors[2].name = "winsor"
        self.outlier_processors[3].name = "trimm"
        self.outlier_processors[4].name = "mad"

        # Interpolators
        self.interpolators = [
            LinearInterpolator(),
            SplineLinearInterpolator(),
            TimeMeanFillInterpolator(),
            MonthlyMeanFillInterpolator(),
            ShiftFillInterpolator(periods=7 * 24),
        ]

        # Set custom names for clarity
        self.interpolators[0].name = "linear"
        self.interpolators[1].name = "spline"
        self.interpolators[2].name = "time_mean"
        self.interpolators[3].name = "monthly_mean"
        self.interpolators[4].name = "week_shift"

    def prepare_data(self):
        """Prepare training and test data"""
        logger.info("Preparing training and test data")

        # Split data into training and test sets
        training_df = self.raw_df.loc[: self.test_start_date]
        test_df = self.raw_df.loc[self.test_start_date : self.test_end_date]

        # Ensure test data has no missing values
        test_data_no_na = test_df.dropna(axis=1)
        test_df = test_data_no_na
        training_df = training_df[test_df.columns]

        logger.info(f"Training data shape: {training_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")

        # Save clean test data for evaluation
        test_df.to_hdf(os.path.join(self.prediction_dir, "test.h5"), key="data")

        self.training_df = training_df
        self.test_df = test_df
        return training_df, test_df

    def create_corrupted_test_data(self):
        """Create test data with artificial outliers and missing values"""
        logger.info(
            "Creating corrupted test data with artificial outliers and missing values"
        )

        # Calculate outlier ratio from training data
        t_all = self.training_df.values.flatten()
        t_all = t_all[~np.isnan(t_all)]
        t_mean = t_all.mean()
        t_std = t_all.std()
        t_z = (t_all - t_mean) / t_std
        t_outlier_indices = np.where(np.abs(t_z) > 3)[0]
        t_outliers = t_all[t_outlier_indices]

        # Calculate ratios
        outlier_ratio = len(t_outliers) / len(t_all)
        missing_ratio = self.training_df.isna().sum().sum() / self.training_df.size

        logger.info(f"Outlier ratio in training data: {outlier_ratio:.4f}")
        logger.info(f"Missing value ratio in training data: {missing_ratio:.4f}")

        # Start with a clean copy of test data
        corrupted_test_df = self.test_df.copy()

        # Add missing values as blocks
        n_missing_per_sensor = int(self.test_df.shape[0] * missing_ratio)

        for col_idx, col in enumerate(self.test_df.columns):
            available_rows = set(range(self.test_df.shape[0]))

            # Create missing blocks of various lengths
            missing_lengths = []
            remaining = n_missing_per_sensor

            while remaining > 0:
                length = min(np.random.randint(1, 5), remaining)  # Max block size of 4
                missing_lengths.append(length)
                remaining -= length

            # Insert missing blocks
            for length in missing_lengths:
                if len(available_rows) < length:
                    break

                potential_starts = [
                    row
                    for row in available_rows
                    if all((row + i) in available_rows for i in range(length))
                ]

                if not potential_starts:
                    continue

                start_row = np.random.choice(potential_starts)
                for i in range(length):
                    row = start_row + i
                    corrupted_test_df.iloc[row, col_idx] = np.nan
                    available_rows.remove(row)

        # Add outliers
        total_points = corrupted_test_df.size
        n_outliers = int(total_points * outlier_ratio)

        # Find valid positions (non-NaN)
        nan_mask = corrupted_test_df.isna().to_numpy()
        valid_indices = np.where(~nan_mask.flatten())[0]

        if len(valid_indices) < n_outliers:
            n_outliers = len(valid_indices)

        # Select positions for outliers
        outlier_positions = np.random.choice(valid_indices, n_outliers, replace=False)

        # Insert outliers
        for pos in outlier_positions:
            row = pos // corrupted_test_df.shape[1]
            col = pos % corrupted_test_df.shape[1]
            outlier_value = np.random.choice(t_outliers)
            corrupted_test_df.iloc[row, col] = outlier_value

        logger.info(
            f"Created corrupted test data with {n_outliers} outliers and {corrupted_test_df.isna().sum().sum()} missing values"
        )

        self.corrupted_test_df = corrupted_test_df
        return corrupted_test_df

    def prepare_evaluation_datasets(self):
        """Prepare datasets for evaluation"""
        logger.info("Preparing evaluation datasets")

        # Simple Replacement Evaluation (srep) - Includes both training and corrupted test data
        srep_df = pd.concat(
            [self.training_df, self.corrupted_test_df], axis=0, copy=True
        )
        srep_df.to_hdf(os.path.join(self.outlier_stest_dir, "base_alt.h5"), key="data")

        # Process srep data with outlier processors
        srep_outlier_paths = remove_outliers(
            data=srep_df,
            outlier_processors=self.outlier_processors,
            output_dir=self.outlier_stest_dir,
        )

        # Training data only for prediction models
        self.training_df.to_hdf(
            os.path.join(self.outlier_ptest_dir, "base_alt.h5"), key="data"
        )

        # Process training data with outlier processors
        pred_outlier_paths = remove_outliers(
            data=self.training_df,
            outlier_processors=self.outlier_processors,
            output_dir=self.outlier_ptest_dir,
        )

        logger.info(
            f"Created {len(srep_outlier_paths)} outlier-processed datasets for simple replacement evaluation"
        )
        logger.info(
            f"Created {len(pred_outlier_paths)} outlier-processed datasets for prediction models"
        )

        return srep_outlier_paths, pred_outlier_paths

    def apply_interpolation(self):
        """Apply interpolation methods to outlier-processed data"""
        logger.info("Applying interpolation methods")

        # Get outlier processed datasets
        srep_outlier_results = [
            TrafficData.import_from_hdf(path)
            for path in [
                os.path.join(self.outlier_stest_dir, f)
                for f in sorted(os.listdir(self.outlier_stest_dir))
                if f.endswith(".h5")
            ]
        ]

        pred_outlier_results = [
            TrafficData.import_from_hdf(path)
            for path in [
                os.path.join(self.outlier_ptest_dir, f)
                for f in sorted(os.listdir(self.outlier_ptest_dir))
                if f.endswith(".h5")
            ]
        ]

        # Apply interpolation to srep datasets
        srep_interpolated_paths = interpolate(
            srep_outlier_results, self.interpolators, self.interpolated_stest_dir
        )

        # Apply interpolation to prediction datasets
        pred_interpolated_paths = interpolate(
            pred_outlier_results, self.interpolators, self.interpolated_ptest_dir
        )

        logger.info(
            f"Created {len(srep_interpolated_paths)} interpolated datasets for simple replacement evaluation"
        )
        logger.info(
            f"Created {len(pred_interpolated_paths)} interpolated datasets for prediction models"
        )

        return srep_interpolated_paths, pred_interpolated_paths

    def evaluate_performance(self):
        """Evaluate performance of different outlier and interpolation combinations"""
        logger.info("Evaluating performance of outlier and interpolation combinations")

        # Load test data (ground truth)
        test_df = pd.read_hdf(os.path.join(self.prediction_dir, "test.h5"))

        # Load srep interpolated results
        srep_results = [
            TrafficData.import_from_hdf(path)
            for path in [
                os.path.join(self.interpolated_stest_dir, f)
                for f in sorted(os.listdir(self.interpolated_stest_dir))
                if f.endswith(".h5")
            ]
        ]

        # Extract test period from interpolated results for comparison
        srep_targets = {
            data.path: data.data.loc[self.test_start_date : self.test_end_date]
            for data in srep_results
        }

        # Calculate metrics for each combination
        metrics_results = {}

        for model_name, pred_df in srep_targets.items():
            # Clean filename for display
            display_name = os.path.basename(model_name)

            # Calculate global metrics
            mask = ~(test_df.isna() | pred_df.isna())
            true_values = test_df.values[mask.values]
            pred_values = pred_df.values[mask.values]

            global_mae = mean_absolute_error(true_values, pred_values)
            global_rmse = root_mean_squared_error(true_values, pred_values)

            # Calculate per-sensor metrics
            per_sensor_metrics = {}
            for col in test_df.columns:
                mask = ~(test_df[col].isna() | pred_df[col].isna())
                if mask.sum() > 0:
                    col_true = test_df.loc[mask, col]
                    col_pred = pred_df.loc[mask, col]
                    mae = mean_absolute_error(col_true, col_pred)
                    rmse = root_mean_squared_error(col_true, col_pred)
                    per_sensor_metrics[col] = {
                        "mae": mae,
                        "rmse": rmse,
                        "count": len(col_true),
                    }

            metrics_results[display_name] = {
                "global": {
                    "mae": global_mae,
                    "rmse": global_rmse,
                    "count": len(true_values),
                },
                "per_sensor": per_sensor_metrics,
            }

            logger.info(
                f"{display_name}: MAE={global_mae:.4f}, RMSE={global_rmse:.4f}, Samples={len(true_values)}"
            )

        # Save results
        self.metrics_results = metrics_results

        # Calculate metrics specifically for outliers
        self._evaluate_outlier_performance(test_df, srep_targets)

        return metrics_results

    def _evaluate_outlier_performance(self, test_df, srep_targets, z_threshold=3.0):
        """Evaluate performance specifically on outlier data points"""
        logger.info("Evaluating performance specifically on outliers")

        outlier_metrics = {}

        for model_name, pred_df in srep_targets.items():
            display_name = os.path.basename(model_name)

            # Identify outliers using z-score
            outlier_mask = pd.DataFrame(
                False, index=test_df.index, columns=test_df.columns
            )

            for col in test_df.columns:
                col_values = test_df[col].dropna()
                if len(col_values) > 0:
                    col_mean = col_values.mean()
                    col_std = col_values.std()
                    if col_std > 0:
                        z_scores = (test_df[col] - col_mean) / col_std
                        outlier_mask[col] = abs(z_scores) > z_threshold

            # Only consider valid (non-NaN) outliers
            valid_mask = ~(test_df.isna() | pred_df.isna())
            combined_mask = outlier_mask & valid_mask

            if not combined_mask.any().any():
                outlier_metrics[display_name] = {"mae": 0.0, "rmse": 0.0, "count": 0}
                continue

            # Extract outlier values
            true_outliers = test_df.values[combined_mask.values]
            pred_outliers = pred_df.values[combined_mask.values]

            # Calculate metrics
            mae = mean_absolute_error(true_outliers, pred_outliers)
            rmse = root_mean_squared_error(true_outliers, pred_outliers)

            outlier_metrics[display_name] = {
                "mae": mae,
                "rmse": rmse,
                "count": len(true_outliers),
            }
            logger.info(
                f"Outliers in {display_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, Count={len(true_outliers)}"
            )

        self.outlier_metrics = outlier_metrics
        return outlier_metrics

    def visualize_results(self):
        """Visualize evaluation results"""
        if not self.metrics_results:
            logger.warning(
                "No metrics results to visualize. Run evaluate_performance first."
            )
            return

        # 1. Global metrics comparison
        self._plot_global_metrics_comparison()

        # 2. Outlier metrics comparison
        self._plot_outlier_metrics_comparison()

        # 3. Per-sensor metrics distribution
        self._plot_per_sensor_metrics_distribution()

        # 4. Generate detailed report
        self._generate_detailed_metrics_report()

    def _plot_global_metrics_comparison(self):
        """Plot comparison of global metrics across methods"""
        logger.info("Plotting global metrics comparison")

        model_names = list(self.metrics_results.keys())
        mae_values = [
            metrics["global"]["mae"] for metrics in self.metrics_results.values()
        ]
        rmse_values = [
            metrics["global"]["rmse"] for metrics in self.metrics_results.values()
        ]

        # Sort by MAE
        sorted_indices = np.argsort(mae_values)
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_mae = [mae_values[i] for i in sorted_indices]
        sorted_rmse = [rmse_values[i] for i in sorted_indices]

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

        # MAE graph
        sns.barplot(x=sorted_names, y=sorted_mae, ax=axs[0], palette="viridis")
        axs[0].set_title("Model MAE Comparison", fontsize=14)
        axs[0].set_ylabel("MAE", fontsize=12)
        axs[0].set_xticklabels(sorted_names, rotation=45, ha="right")
        axs[0].set_ylim(bottom=0)

        # RMSE graph
        sns.barplot(x=sorted_names, y=sorted_rmse, ax=axs[1], palette="viridis")
        axs[1].set_title("Model RMSE Comparison", fontsize=14)
        axs[1].set_ylabel("RMSE", fontsize=12)
        axs[1].set_xticklabels(sorted_names, rotation=45, ha="right")
        axs[1].set_ylim(bottom=0)

        plt.suptitle("Comparison of Methods by Global Metrics", fontsize=16, y=1.05)
        plt.tight_layout()

        # Add values above bars
        for i, ax in enumerate([axs[0], axs[1]]):
            values = sorted_mae if i == 0 else sorted_rmse
            for j, v in enumerate(values):
                ax.text(j, v + v * 0.01, f"{v:.3f}", ha="center", fontsize=9)

        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, "global_metrics_comparison.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_outlier_metrics_comparison(self):
        """Plot comparison of outlier-specific metrics across methods"""
        if not self.outlier_metrics:
            logger.warning("No outlier metrics to visualize")
            return

        logger.info("Plotting outlier metrics comparison")

        # Prepare data
        model_names = list(self.outlier_metrics.keys())
        mae_values = [metrics["mae"] for metrics in self.outlier_metrics.values()]
        rmse_values = [metrics["rmse"] for metrics in self.outlier_metrics.values()]

        # Sort by MAE
        sorted_indices = np.argsort(mae_values)
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_mae = [mae_values[i] for i in sorted_indices]
        sorted_rmse = [rmse_values[i] for i in sorted_indices]

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

        # MAE graph
        sns.barplot(x=sorted_names, y=sorted_mae, ax=axs[0], palette="plasma")
        axs[0].set_title("Outlier MAE Comparison", fontsize=14)
        axs[0].set_ylabel("MAE", fontsize=12)
        axs[0].set_xticklabels(sorted_names, rotation=45, ha="right")
        axs[0].set_ylim(bottom=0)

        # RMSE graph
        sns.barplot(x=sorted_names, y=sorted_rmse, ax=axs[1], palette="plasma")
        axs[1].set_title("Outlier RMSE Comparison", fontsize=14)
        axs[1].set_ylabel("RMSE", fontsize=12)
        axs[1].set_xticklabels(sorted_names, rotation=45, ha="right")
        axs[1].set_ylim(bottom=0)

        plt.suptitle("Comparison of Methods for Outlier Handling", fontsize=16, y=1.05)
        plt.tight_layout()

        # Add values above bars
        for i, ax in enumerate([axs[0], axs[1]]):
            values = sorted_mae if i == 0 else sorted_rmse
            for j, v in enumerate(values):
                ax.text(j, v + v * 0.01, f"{v:.3f}", ha="center", fontsize=9)

        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, "outlier_metrics_comparison.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_per_sensor_metrics_distribution(self):
        """Plot distribution of per-sensor metrics"""
        logger.info("Plotting per-sensor metrics distribution")

        # Collect per-sensor MAE and RMSE for each method
        method_sensor_metrics = {}

        for method_name, metrics in self.metrics_results.items():
            sensor_mae = [
                sensor_metric["mae"] for sensor_metric in metrics["per_sensor"].values()
            ]
            sensor_rmse = [
                sensor_metric["rmse"]
                for sensor_metric in metrics["per_sensor"].values()
            ]
            method_sensor_metrics[method_name] = {
                "mae": sensor_mae,
                "rmse": sensor_rmse,
            }

        # Select top 5 methods based on median MAE
        median_maes = [
            (name, np.median(data["mae"]))
            for name, data in method_sensor_metrics.items()
        ]
        top_methods = sorted(median_maes, key=lambda x: x[1])[:5]
        top_method_names = [m[0] for m in top_methods]

        # Create box plots
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

        # Prepare data for box plots
        mae_data = []
        mae_labels = []
        rmse_data = []
        rmse_labels = []

        for method in top_method_names:
            mae_data.extend(method_sensor_metrics[method]["mae"])
            mae_labels.extend([method] * len(method_sensor_metrics[method]["mae"]))

            rmse_data.extend(method_sensor_metrics[method]["rmse"])
            rmse_labels.extend([method] * len(method_sensor_metrics[method]["rmse"]))

        # Create DataFrames for seaborn
        mae_df = pd.DataFrame({"Method": mae_labels, "MAE": mae_data})
        rmse_df = pd.DataFrame({"Method": rmse_labels, "RMSE": rmse_data})

        # Plot
        sns.boxplot(x="Method", y="MAE", data=mae_df, ax=axs[0], palette="viridis")
        axs[0].set_title("Per-Sensor MAE Distribution (Top 5 Methods)", fontsize=14)
        axs[0].set_ylabel("MAE", fontsize=12)
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

        sns.boxplot(x="Method", y="RMSE", data=rmse_df, ax=axs[1], palette="viridis")
        axs[1].set_title("Per-Sensor RMSE Distribution (Top 5 Methods)", fontsize=14)
        axs[1].set_ylabel("RMSE", fontsize=12)
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")

        plt.suptitle("Distribution of Per-Sensor Metrics", fontsize=16, y=1.05)
        plt.tight_layout()

        # Save figure
        plt.savefig(
            os.path.join(self.output_dir, "per_sensor_metrics_distribution.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_detailed_metrics_report(self):
        """Generate detailed metrics report"""
        logger.info("Generating detailed metrics report")

        # Create DataFrame for global metrics
        global_metrics = {"Method": [], "MAE": [], "RMSE": [], "Sample Count": []}

        for method, metrics in self.metrics_results.items():
            global_metrics["Method"].append(method)
            global_metrics["MAE"].append(metrics["global"]["mae"])
            global_metrics["RMSE"].append(metrics["global"]["rmse"])
            global_metrics["Sample Count"].append(metrics["global"]["count"])

        global_df = pd.DataFrame(global_metrics)
        global_df = global_df.sort_values("MAE")

        # Create DataFrame for outlier metrics
        outlier_metrics = {
            "Method": [],
            "Outlier MAE": [],
            "Outlier RMSE": [],
            "Outlier Count": [],
        }

        for method, metrics in self.outlier_metrics.items():
            outlier_metrics["Method"].append(method)
            outlier_metrics["Outlier MAE"].append(metrics["mae"])
            outlier_metrics["Outlier RMSE"].append(metrics["rmse"])
            outlier_metrics["Outlier Count"].append(metrics["count"])

        outlier_df = pd.DataFrame(outlier_metrics)
        outlier_df = outlier_df.sort_values("Outlier MAE")

        # Create per-sensor metrics
        sensor_metrics_by_method = {}

        for method, metrics in self.metrics_results.items():
            for sensor, sensor_metric in metrics["per_sensor"].items():
                if sensor not in sensor_metrics_by_method:
                    sensor_metrics_by_method[sensor] = {}
                sensor_metrics_by_method[sensor][f"{method}_MAE"] = sensor_metric["mae"]
                sensor_metrics_by_method[sensor][f"{method}_RMSE"] = sensor_metric[
                    "rmse"
                ]

        sensor_df = pd.DataFrame(sensor_metrics_by_method).T

        # Save to Excel
        with pd.ExcelWriter(
            os.path.join(self.output_dir, "detailed_metrics_report.xlsx")
        ) as writer:
            global_df.to_excel(writer, sheet_name="Global Metrics", index=False)
            outlier_df.to_excel(writer, sheet_name="Outlier Metrics", index=False)
            sensor_df.to_excel(writer, sheet_name="Per Sensor Metrics")

        logger.info(
            f"Detailed report saved to {os.path.join(self.output_dir, 'detailed_metrics_report.xlsx')}"
        )

    def export_best_model_configs(self):
        """Export configurations of the best performing models"""
        if not self.metrics_results:
            logger.warning(
                "No metrics results available. Run evaluate_performance first."
            )
            return

        logger.info("Exporting best model configurations")

        # Find best models based on different criteria
        model_configs = {
            "best_overall_mae": min(
                self.metrics_results.items(), key=lambda x: x[1]["global"]["mae"]
            )[0],
            "best_overall_rmse": min(
                self.metrics_results.items(), key=lambda x: x[1]["global"]["rmse"]
            )[0],
            "best_outlier_mae": (
                min(self.outlier_metrics.items(), key=lambda x: x[1]["mae"])[0]
                if self.outlier_metrics
                else None
            ),
            "best_outlier_rmse": (
                min(self.outlier_metrics.items(), key=lambda x: x[1]["rmse"])[0]
                if self.outlier_metrics
                else None
            ),
        }

        # Parse outlier and interpolation methods from filenames
        best_configs = {}

        for criterion, model_name in model_configs.items():
            if model_name is None:
                continue

            parts = model_name.split("-")
            if len(parts) >= 2:
                outlier_method = parts[0]
                interpolation_method = (
                    parts[1].split(".")[0] if "." in parts[1] else parts[1]
                )

                best_configs[criterion] = {
                    "model_name": model_name,
                    "outlier_method": outlier_method,
                    "interpolation_method": interpolation_method,
                }

        # Save configurations
        with open(os.path.join(self.output_dir, "best_model_configs.txt"), "w") as f:
            f.write("Best Model Configurations\n")
            f.write("========================\n\n")

            for criterion, config in best_configs.items():
                f.write(f"{criterion.replace('_', ' ').title()}:\n")
                f.write(f"  Model: {config['model_name']}\n")
                f.write(f"  Outlier Method: {config['outlier_method']}\n")
                f.write(f"  Interpolation Method: {config['interpolation_method']}\n\n")

        logger.info(
            f"Best model configurations exported to {os.path.join(self.output_dir, 'best_model_configs.txt')}"
        )
        return best_configs


def run_evaluation(args):
    """Run full evaluation workflow"""
    evaluator = OutlierAndInterpolationEvaluator(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
    )

    # Run full workflow
    evaluator.prepare_data()
    evaluator.create_corrupted_test_data()
    evaluator.prepare_evaluation_datasets()
    evaluator.apply_interpolation()
    evaluator.evaluate_performance()
    evaluator.visualize_results()
    evaluator.export_best_model_configs()

    logger.info("Evaluation completed successfully")