import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging


@dataclass
class ErrorMetrics:
    """에러 메트릭을 저장하는 데이터클래스"""
    mae: float
    rmse: float
    mape: float
    r2: float
    sensor_name: str
    data_points: int


@dataclass
class ErrorCase:
    """개별 에러 케이스를 저장하는 데이터클래스"""
    sensor_name: str
    timestamp: pd.Timestamp
    target_value: float
    predicted_value: float
    absolute_error: float
    relative_error: float


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """기본 회귀 메트릭들을 계산합니다."""
    # NaN 값 제거
    mask = ~(np.isnan(target) | np.isnan(prediction))
    target_clean = target[mask]
    pred_clean = prediction[mask]
    
    if len(target_clean) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(target_clean - pred_clean))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((target_clean - pred_clean) ** 2))
    
    # MAPE (Mean Absolute Percentage Error)
    # 0으로 나누는 것을 방지하기 위해 작은 값을 더함
    mape = np.mean(np.abs((target_clean - pred_clean) / (target_clean + 1e-8))) * 100
    
    # R² (결정계수)
    ss_res = np.sum((target_clean - pred_clean) ** 2)
    ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def analyze_dataset_errors(
    result: Dict[str, pd.DataFrame], 
    dataset_name: str
) -> Tuple[List[ErrorMetrics], List[ErrorCase], go.Figure]:
    """
    데이터셋의 예측 에러를 종합적으로 분석합니다.
    
    Args:
        result: 센서별 예측 결과 딕셔너리
        dataset_name: 데이터셋 이름 (training, validation, test)
    
    Returns:
        sensor_metrics: 센서별 에러 메트릭 리스트
        top_errors: 가장 큰 에러 Top 10 리스트
        analysis_fig: 에러 분석 시각화 Figure
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=== {dataset_name.upper()} 데이터셋 에러 분석 시작 ===")
    
    sensor_metrics = []
    all_errors = []
    
    # 센서별 메트릭 계산
    for sensor_name, df in result.items():
        target = df['target'].values
        prediction = df['prediction'].values
        
        # 메트릭 계산
        metrics = calculate_metrics(target, prediction)
        
        sensor_metric = ErrorMetrics(
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            r2=metrics['r2'],
            sensor_name=sensor_name,
            data_points=len(df)
        )
        sensor_metrics.append(sensor_metric)
        
        # 개별 에러 케이스 수집
        absolute_errors = np.abs(target - prediction)
        relative_errors = np.abs((target - prediction) / (target + 1e-8)) * 100
        
        for i, (time, tar, pred, abs_err, rel_err) in enumerate(zip(
            df['time'], target, prediction, absolute_errors, relative_errors
        )):
            error_case = ErrorCase(
                sensor_name=sensor_name,
                timestamp=time,
                target_value=tar,
                predicted_value=pred,
                absolute_error=abs_err,
                relative_error=rel_err
            )
            all_errors.append(error_case)
    
    # Top 10 에러 케이스 추출
    top_errors = sorted(all_errors, key=lambda x: x.absolute_error, reverse=True)[:10]
    
    # 전체 메트릭 요약
    logger.info("📊 전체 성능 요약:")
    valid_metrics = [m for m in sensor_metrics if not np.isnan(m.mae)]
    if valid_metrics:
        avg_mae = np.mean([m.mae for m in valid_metrics])
        avg_rmse = np.mean([m.rmse for m in valid_metrics])
        avg_mape = np.mean([m.mape for m in valid_metrics])
        avg_r2 = np.mean([m.r2 for m in valid_metrics])
        
        logger.info(f"  평균 MAE: {avg_mae:.4f}")
        logger.info(f"  평균 RMSE: {avg_rmse:.4f}")
        logger.info(f"  평균 MAPE: {avg_mape:.2f}%")
        logger.info(f"  평균 R²: {avg_r2:.4f}")
    
    # 센서별 성능 출력
    logger.info("📈 센서별 성능 (MAE 기준 상위 5개):")
    best_sensors = sorted(valid_metrics, key=lambda x: x.mae)[:5]
    for i, metric in enumerate(best_sensors, 1):
        logger.info(f"  {i}. 센서 {metric.sensor_name}: MAE={metric.mae:.4f}, RMSE={metric.rmse:.4f}")
    
    logger.info("📉 센서별 성능 (MAE 기준 하위 5개):")
    worst_sensors = sorted(valid_metrics, key=lambda x: x.mae, reverse=True)[:5]
    for i, metric in enumerate(worst_sensors, 1):
        logger.info(f"  {i}. 센서 {metric.sensor_name}: MAE={metric.mae:.4f}, RMSE={metric.rmse:.4f}")
    
    # Top 10 에러 케이스 출력
    logger.info("🔥 가장 큰 에러 Top 10:")
    for i, error in enumerate(top_errors, 1):
        logger.info(f"  {i}. 센서 {error.sensor_name} ({error.timestamp})")
        logger.info(f"     실제값: {error.target_value:.2f}, 예측값: {error.predicted_value:.2f}")
        logger.info(f"     절대 에러: {error.absolute_error:.2f}, 상대 에러: {error.relative_error:.1f}%")
    
    # 시각화 생성
    analysis_fig = create_error_analysis_plot(sensor_metrics, top_errors, dataset_name)
    
    return sensor_metrics, top_errors, analysis_fig


def create_error_analysis_plot(
    sensor_metrics: List[ErrorMetrics], 
    top_errors: List[ErrorCase], 
    dataset_name: str
) -> go.Figure:
    """에러 분석 결과를 시각화합니다."""
    
    # 서브플롯 생성 (2x2 레이아웃)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "센서별 MAE 분포",
            "센서별 RMSE vs MAE",
            "Top 10 에러 케이스",
            "에러 메트릭 상관관계"
        ],
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 유효한 메트릭만 필터링
    valid_metrics = [m for m in sensor_metrics if not np.isnan(m.mae)]
    
    if not valid_metrics:
        fig.add_annotation(
            text="유효한 메트릭 데이터가 없습니다.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # 1. MAE 분포 히스토그램
    mae_values = [m.mae for m in valid_metrics]
    fig.add_trace(
        go.Histogram(x=mae_values, name="MAE 분포", nbinsx=20),
        row=1, col=1
    )
    
    # 2. RMSE vs MAE 산점도
    rmse_values = [m.rmse for m in valid_metrics]
    sensor_names = [m.sensor_name for m in valid_metrics]
    fig.add_trace(
        go.Scatter(
            x=mae_values, y=rmse_values,
            mode='markers',
            text=sensor_names,
            name="센서별 성능",
            hovertemplate="<b>센서 %{text}</b><br>MAE: %{x:.4f}<br>RMSE: %{y:.4f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # 3. Top 10 에러 케이스 바차트
    if top_errors:
        error_labels = [f"센서 {e.sensor_name}<br>{e.timestamp.strftime('%m-%d %H:%M')}" 
                       for e in top_errors]
        error_values = [e.absolute_error for e in top_errors]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(error_values))),
                y=error_values,
                text=error_labels,
                name="Top 10 에러",
                hovertemplate="<b>%{text}</b><br>절대 에러: %{y:.2f}<extra></extra>"
            ),
            row=2, col=1
        )
    
    # 4. MAPE vs R² 산점도
    mape_values = [m.mape for m in valid_metrics if not np.isnan(m.mape)]
    r2_values = [m.r2 for m in valid_metrics if not np.isnan(m.r2)]
    
    if len(mape_values) == len(r2_values) and len(mape_values) > 0:
        fig.add_trace(
            go.Scatter(
                x=mape_values, y=r2_values,
                mode='markers',
                text=sensor_names[:len(mape_values)],
                name="MAPE vs R²",
                hovertemplate="<b>센서 %{text}</b><br>MAPE: %{x:.2f}%<br>R²: %{y:.4f}<extra></extra>"
            ),
            row=2, col=2
        )
    
    # 레이아웃 업데이트
    fig.update_layout(
        title=f"{dataset_name.upper()} 데이터셋 에러 분석",
        height=800,
        showlegend=False
    )
    
    # 축 라벨 설정
    fig.update_xaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="빈도", row=1, col=1)
    
    fig.update_xaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    
    fig.update_xaxes(title_text="에러 순위", row=2, col=1)
    fig.update_yaxes(title_text="절대 에러", row=2, col=1)
    
    fig.update_xaxes(title_text="MAPE (%)", row=2, col=2)
    fig.update_yaxes(title_text="R²", row=2, col=2)
    
    return fig


def save_error_analysis_results(
    sensor_metrics: List[ErrorMetrics], 
    top_errors: List[ErrorCase], 
    dataset_name: str, 
    output_dir: str
) -> None:
    """에러 분석 결과를 CSV 파일로 저장합니다."""
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 센서별 메트릭 저장
    metrics_data = []
    for metric in sensor_metrics:
        metrics_data.append({
            'sensor_name': metric.sensor_name,
            'mae': metric.mae,
            'rmse': metric.rmse,
            'mape': metric.mape,
            'r2': metric.r2,
            'data_points': metric.data_points
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = output_path / f"error_metrics_{dataset_name}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"센서별 메트릭 저장: {metrics_path}")
    
    # Top 에러 케이스 저장
    errors_data = []
    for error in top_errors:
        errors_data.append({
            'sensor_name': error.sensor_name,
            'timestamp': error.timestamp,
            'target_value': error.target_value,
            'predicted_value': error.predicted_value,
            'absolute_error': error.absolute_error,
            'relative_error': error.relative_error
        })
    
    errors_df = pd.DataFrame(errors_data)
    errors_path = output_path / f"top_errors_{dataset_name}.csv"
    errors_df.to_csv(errors_path, index=False)
    logger.info(f"Top 10 에러 케이스 저장: {errors_path}")
