from typing import Optional, Tuple, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import pandas as pd

from .model import LSTMBaseModel
from ...utils import set_random_seed  # 추가: 시드 고정 함수 임포트


class LSTMLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training LSTMBase model for traffic prediction
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        seed: Optional[int] = None,  # 수정: 기본값 None, 매번 다른 시드
        gradient_clip_val: float = 1.0,  # 추가: 그래디언트 클리핑 값
        scaler=None,  # 추가: 테스트 메트릭 계산을 위한 스케일러
    ):
        super().__init__()
        # 랜덤 시드 고정 (재현성 확보)
        self.seed = set_random_seed(seed)
        self.save_hyperparameters(ignore=["scaler"])

        # Store scaler for inverse transform during test
        self.scaler = scaler

        # 그래디언트 클리핑 값 저장
        self.gradient_clip_val = gradient_clip_val

        # Initialize the LSTMBase model
        self.model = LSTMBaseModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout_rate=dropout_rate,
        )
        # Loss function declaration
        self.criterion = nn.MSELoss()

        # Learning rate and scheduler parameters
        self.learning_rate = learning_rate
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step"""
        x, y = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)  # Remove last dimension if it's 1

        loss: torch.Tensor = self.criterion(y_hat, y)

        # 그래디언트 클리핑 적용 (폭발 방지)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
        self.validation_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step"""
        x, y = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch"""
        if len(self.validation_outputs) == 0:
            return

        # Concatenate all predictions and targets
        y_true = np.concatenate([x["y_true"] for x in self.validation_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.validation_outputs], axis=0)

        # Calculate metrics on scaled data
        mae_scaled = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse_scaled = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

        self.log("val_mae", mae_scaled)
        self.log("val_rmse", rmse_scaled)

        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch.

        Metrics are computed on the original (unscaled) data using inverse transform.
        If is_missing information is available, only original (non-interpolated) data is used.
        """
        if len(self.test_outputs) == 0:
            return

        # Check if is_missing information is available
        has_missing_info = "is_missing" in self.test_outputs[0]
        
        if has_missing_info:
            # Filter to only include original (non-interpolated) data
            original_outputs = []
            for output in self.test_outputs:
                is_missing_list = output["is_missing"]
                y_true = output["y_true"]
                y_pred = output["y_pred"]
                
                # Filter out interpolated values
                original_mask = ~np.array(is_missing_list, dtype=bool)
                if original_mask.any():
                    original_outputs.append({
                        "y_true": y_true[original_mask],
                        "y_pred": y_pred[original_mask],
                    })
            
            if len(original_outputs) == 0:
                print("\nWarning: No original (non-interpolated) test data found!")
                self.test_outputs.clear()
                return
            
            # Concatenate filtered predictions and targets
            y_true = np.concatenate([x["y_true"] for x in original_outputs], axis=0)
            y_pred = np.concatenate([x["y_pred"] for x in original_outputs], axis=0)
            
            total_samples = sum(len(output["is_missing"]) for output in self.test_outputs)
            original_samples = len(y_true)
            print(f"\nUsing {original_samples}/{total_samples} original samples (excluding {total_samples - original_samples} interpolated values)")
        else:
            # No missing info, use all data
            y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
            y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)

        # Calculate scaled metrics first (before inverse transform)
        y_true_scaled_flat = y_true.flatten()
        y_pred_scaled_flat = y_pred.flatten()
        
        mae_scaled = mean_absolute_error(y_true_scaled_flat, y_pred_scaled_flat)
        rmse_scaled = np.sqrt(mean_squared_error(y_true_scaled_flat, y_pred_scaled_flat))
        
        non_zero_mask_scaled = y_true_scaled_flat != 0
        mape_scaled = (
            np.mean(np.abs((y_true_scaled_flat[non_zero_mask_scaled] - y_pred_scaled_flat[non_zero_mask_scaled]) / y_true_scaled_flat[non_zero_mask_scaled]))
            * 100
            if non_zero_mask_scaled.any()
            else 0.0
        )
        
        # Log scaled metrics
        self.log("test_mae_scaled", mae_scaled)
        self.log("test_rmse_scaled", rmse_scaled)
        self.log("test_mape_scaled", float(mape_scaled))
        
        print(f"\nTest Results (Scaled):")
        print(f"MAE: {mae_scaled:.4f}")
        print(f"RMSE: {rmse_scaled:.4f}")
        print(f"MAPE: {mape_scaled:.2f}%")

        # Inverse transform to get original scale values
        if self.scaler is not None:
            # Store original shapes
            original_shape = y_true.shape

            # Flatten, inverse transform, and reshape back
            y_true_flat = y_true.reshape(-1, 1)
            y_pred_flat = y_pred.reshape(-1, 1)

            y_true_unscaled = self.scaler.inverse_transform(y_true_flat).reshape(
                original_shape
            )
            y_pred_unscaled = self.scaler.inverse_transform(y_pred_flat).reshape(
                original_shape
            )
        else:
            # No scaler provided, use original values
            y_true_unscaled = y_true
            y_pred_unscaled = y_pred

        # Calculate metrics on unscaled (original) data
        mae = mean_absolute_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten())
        rmse = np.sqrt(
            mean_squared_error(y_true_unscaled.flatten(), y_pred_unscaled.flatten())
        )

        # MAPE with zero-division handling
        y_true_flat = y_true_unscaled.flatten()
        y_pred_flat = y_pred_unscaled.flatten()
        mask = y_true_flat != 0
        mape = (
            np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]))
            * 100
            if mask.any()
            else 0.0
        )

        self.log("test_mae", mae)
        self.log("test_rmse", rmse)
        self.log("test_mape", float(mape))

        print(f"\nTest Results (Original Scale):")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        # Clear outputs
        self.test_outputs.clear()


# --------------------------------------------------------------------


class MultiSensorLSTMLightningModule(LSTMLightningModule):
    """
    MultiSensorTrafficDataModule과 collate_multi_sensor를 사용하는 LSTM Lightning Module.
    센서별 정보를 포함한 배치 데이터를 처리하고 센서별 성능 추적 가능.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        track_sensor_metrics: bool = True,
        seed: Optional[int] = None,  # 수정: 기본값 None, 매번 다른 시드
        gradient_clip_val: float = 1.0,  # 추가: 그래디언트 클리핑 값
        scaler=None,  # 추가: 테스트 메트릭 계산을 위한 스케일러
    ):
        """
        Args:
            track_sensor_metrics: 센서별 메트릭 추적 여부
            seed: 랜덤 시드 (재현성)
            gradient_clip_val: 그래디언트 클리핑 값
            scaler: 테스트 메트릭 계산을 위한 스케일러 (inverse_transform용)
            기타 매개변수는 부모 클래스와 동일
        """
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            seed=seed,
            gradient_clip_val=gradient_clip_val,
            scaler=scaler,
        )

        self.track_sensor_metrics = track_sensor_metrics

        # 센서별 메트릭 저장을 위한 추가 저장소
        self.validation_sensor_outputs = defaultdict(list)
        self.test_sensor_outputs = defaultdict(list)

    def training_step(
        self,
        batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            List[pd.DatetimeIndex],
            List[pd.Timestamp],
            List[str],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step - collate_multi_sensor 배치 형태 처리"""
        x, y, x_time_indices, y_time_indices, sensor_names = batch
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # 그래디언트 클리핑 적용 (폭발 방지)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        # 센서별 정보를 로깅 (선택적)
        if (
            self.track_sensor_metrics and batch_idx % 100 == 0
        ):  # 100번째 배치마다만 로깅
            unique_sensors = set(sensor_names)
            self.log(
                "train_unique_sensors",
                len(unique_sensors),
                on_step=False,
                on_epoch=True,
            )

        return loss

    def validation_step(
        self,
        batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            List[pd.DatetimeIndex],
            List[pd.Timestamp],
            List[str],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step - 센서별 정보 포함 처리"""
        x, y, x_time_indices, y_time_indices, sensor_names = batch
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # 기본 검증 출력 저장 (부모 클래스와 동일)
        self.validation_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        # 센서별 출력 저장 (선택적)
        if self.track_sensor_metrics:
            y_np = y.cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()

            for i, sensor_name in enumerate(sensor_names):
                self.validation_sensor_outputs[sensor_name].append(
                    {
                        "y_true": y_np[i : i + 1],  # 개별 샘플
                        "y_pred": y_hat_np[i : i + 1],
                        "loss": self.criterion(y_hat[i : i + 1], y[i : i + 1]).item(),
                    }
                )

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(
        self,
        batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            List[pd.DatetimeIndex],
            List[pd.Timestamp],
            List[str],
            List[bool],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step - 센서별 정보 포함 처리"""
        x, y, x_time_indices, y_time_indices, sensor_names, y_is_missing_list = batch
        y_hat: torch.Tensor = self(x)

        # Reshape y if needed
        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # 기본 테스트 출력 저장 (is_missing 정보 포함)
        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
                "is_missing": y_is_missing_list,  # missing 정보 저장
            }
        )

        # 센서별 출력 저장 (선택적)
        if self.track_sensor_metrics:
            y_np = y.cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()

            for i, sensor_name in enumerate(sensor_names):
                self.test_sensor_outputs[sensor_name].append(
                    {
                        "y_true": y_np[i : i + 1],
                        "y_pred": y_hat_np[i : i + 1],
                        "loss": self.criterion(y_hat[i : i + 1], y[i : i + 1]).item(),
                        "is_missing": y_is_missing_list[i],  # missing 정보 저장
                    }
                )

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """검증 에포크 종료 - 전체 메트릭만 계산"""
        # 부모 클래스의 전체 메트릭 계산 실행
        super().on_validation_epoch_end()

        # 센서별 출력 데이터 정리 (로깅 없이)
        if self.track_sensor_metrics:
            self.validation_sensor_outputs.clear()

    def on_test_epoch_end(self):
        """테스트 에포크 종료 - 전체 메트릭만 계산"""
        # 부모 클래스의 전체 메트릭 계산 실행
        super().on_test_epoch_end()

        # 센서별 메트릭 출력 (로깅 없이, 콘솔 출력만)
        if self.track_sensor_metrics and len(self.test_sensor_outputs) > 0:
            self._print_sensor_summary()
            # 메모리 효율성을 위해 센서별 데이터 정리
            self.test_sensor_outputs.clear()

    def _print_sensor_summary(self):
        """센서별 테스트 메트릭 요약 출력 (상위 5개, 하위 5개만) - 원본 데이터만 사용"""
        if len(self.test_sensor_outputs) == 0:
            return

        sensor_metrics = {}

        # 각 센서별 메트릭 계산 (보간되지 않은 원본 데이터만 사용)
        for sensor_name, outputs in self.test_sensor_outputs.items():
            if len(outputs) == 0:
                continue

            # 원본 데이터만 필터링 (is_missing == False)
            original_outputs = [x for x in outputs if not x.get("is_missing", False)]

            if len(original_outputs) == 0:
                continue  # 원본 데이터가 없으면 스킵

            y_true = np.concatenate([x["y_true"] for x in original_outputs], axis=0)
            y_pred = np.concatenate([x["y_pred"] for x in original_outputs], axis=0)

            mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

            total_samples = len(outputs)
            original_samples = len(original_outputs)

            sensor_metrics[sensor_name] = {
                "mae": mae,
                "rmse": rmse,
                "original_samples": original_samples,
                "total_samples": total_samples,
            }

        if len(sensor_metrics) == 0:
            return

        # MAE 기준으로 정렬
        sorted_sensors = sorted(sensor_metrics.items(), key=lambda x: x[1]["mae"])

        print(f"\n{'='*70}")
        print(f"Sensor Performance Summary ({len(sensor_metrics)} sensors)")
        print(f"Only using ORIGINAL data (excluding interpolated values)")
        print(f"{'='*70}")

        # 전체 통계
        all_maes = [metrics["mae"] for metrics in sensor_metrics.values()]
        all_rmses = [metrics["rmse"] for metrics in sensor_metrics.values()]
        total_original = sum(m["original_samples"] for m in sensor_metrics.values())
        total_all = sum(m["total_samples"] for m in sensor_metrics.values())

        print(f"Overall Statistics:")
        print(f"  MAE - Mean: {np.mean(all_maes):.4f}, Std: {np.std(all_maes):.4f}")
        print(f"  RMSE - Mean: {np.mean(all_rmses):.4f}, Std: {np.std(all_rmses):.4f}")
        print(
            f"  Samples - Original: {total_original}, Total: {total_all} ({total_original/total_all*100:.1f}% original)"
        )
        print()

        # 최고 성능 5개
        print("Top 5 Best Performing Sensors (by MAE):")
        for i, (sensor_name, metrics) in enumerate(sorted_sensors[:5]):
            print(
                f"  {i+1}. {sensor_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f} "
                f"({metrics['original_samples']}/{metrics['total_samples']} samples)"
            )

        print()

        # 최저 성능 5개
        print("Top 5 Worst Performing Sensors (by MAE):")
        for i, (sensor_name, metrics) in enumerate(sorted_sensors[-5:]):
            rank = len(sorted_sensors) - 4 + i
            print(
                f"  {rank}. {sensor_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f} "
                f"({metrics['original_samples']}/{metrics['total_samples']} samples)"
            )

        print(f"{'='*70}")
