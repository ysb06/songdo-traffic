import pandas as pd
import plotly.graph_objects as go
import torch
from metr.datasets.rnn.datamodule import MultiSensorTrafficDataModule
from tqdm import tqdm
import lightning.pytorch as L
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, Union
import logging

from .utils import load_or_train_model, sanitize_sensor_name
from . import MODEL_OUTPUT_DIR


def analyze_predictions(
    dataloader: torch.utils.data.DataLoader,
    model: L.LightningModule,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Tuple[go.Figure, Dict[str, pd.DataFrame]]:
    """데이터로더의 모든 배치에 대해 모델 예측을 수행하고, 센서별 예측 결과를 시각화합니다."""
    logger = logging.getLogger(__name__)
    result = {}

    # training_loader를 순회하며 예측값 얻기
    logger.info("모델 예측 시작...")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            # 배치 데이터 추출
            xs: torch.Tensor = batch[0]
            ys: torch.Tensor = batch[1]
            y_time_indices: list[pd.DatetimeIndex] = batch[3]
            sensor_names: list[str] = batch[4]

            # 입력 데이터를 모델과 같은 디바이스로 이동
            xs = xs.to(device)
            ys = ys.to(device)

            # 모델을 통한 예측
            pred: torch.Tensor = model(xs)

            # 각 센서별로 예측 결과를 데이터프레임으로 저장
            for i, sensor_name in enumerate(sensor_names):
                # 시간 인덱스 (예측 시점)
                time_idx = y_time_indices[i]

                # 실제값과 예측값 (배치의 i번째 샘플)
                target_values = ys[i].cpu().numpy().flatten()  # (sequence_length,)
                pred_values = pred[i].cpu().numpy().flatten()  # (sequence_length,)

                # 데이터프레임 생성
                df = pd.DataFrame(
                    {
                        "time": time_idx,
                        "target": target_values,
                        "prediction": pred_values,
                    }
                )

                # 센서별 결과를 딕셔너리에 저장 (기존 데이터에 추가)
                if sensor_name in result:
                    result[sensor_name] = pd.concat(
                        [result[sensor_name], df], ignore_index=True
                    )
                else:
                    result[sensor_name] = df

    logger.info("예측 완료!")
    logger.info(f"처리된 센서 수: {len(result)}")
    for sensor_name, df in result.items():
        logger.debug(f"센서 {sensor_name}: {len(df)}개 데이터 포인트")

    # 결과 저장 (선택사항)
    if save_path:
        save_prediction_results(result, save_path)
        logger.info(f"예측 결과 저장 완료: {save_path}")

    # 시각화 생성
    logger.info("시각화 생성 중...")
    fig = create_interactive_plot(result)

    return fig, result


def save_prediction_results(result: Dict[str, pd.DataFrame], save_path: str) -> None:
    """예측 결과를 HDF5와 pickle 형식으로 저장합니다."""
    logger = logging.getLogger(__name__)
    save_path: Path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # HDF5로 주요 데이터 저장
    h5_path = save_path.with_suffix(".h5")
    with pd.HDFStore(h5_path, mode="w", complevel=9, complib="zlib") as store:
        for sensor_name, df in result.items():
            # 센서명을 안전한 키로 변환
            safe_key = sanitize_sensor_name(sensor_name)
            store[safe_key] = df

    # 메타데이터를 pickle로 저장
    metadata = {
        "sensor_mapping": {
            sanitize_sensor_name(sensor_name): sensor_name
            for sensor_name in result.keys()
        },
        "total_sensors": len(result),
        "data_points_per_sensor": {sensor: len(df) for sensor, df in result.items()},
        "time_range": {
            "start": min(df["time"].min() for df in result.values()),
            "end": max(df["time"].max() for df in result.values()),
        },
    }

    pickle_path = save_path.with_suffix(".pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(metadata, f)
        
    logger.debug(f"HDF5 파일 저장: {h5_path}")
    logger.debug(f"메타데이터 저장: {pickle_path}")


def analyze_predictions_from_file(
    load_path: str,
) -> Tuple[go.Figure, Dict[str, pd.DataFrame]]:
    """저장된 예측 결과 파일로부터 시각화를 생성합니다."""
    logger = logging.getLogger(__name__)
    load_path: Path = Path(load_path)

    # 파일 존재 확인
    h5_path = load_path.with_suffix(".h5")
    pickle_path = load_path.with_suffix(".pkl")

    if not h5_path.exists() or not pickle_path.exists():
        raise FileNotFoundError(
            f"저장된 파일을 찾을 수 없습니다: {h5_path} 또는 {pickle_path}"
        )

    logger.info(f"예측 결과 로딩: {load_path}")

    # 메타데이터 로드
    with open(pickle_path, "rb") as f:
        metadata: Dict[str, Union[str, int, Dict[str, str]]] = pickle.load(f)

    logger.info(f"총 {metadata['total_sensors']}개 센서 데이터")
    logger.info(f"기간: {metadata['time_range']['start']} ~ {metadata['time_range']['end']}")

    # HDF5에서 데이터 로드
    result = {}
    with pd.HDFStore(h5_path, mode="r") as store:
        for safe_key, original_sensor in metadata["sensor_mapping"].items():
            df = store[safe_key]
            result[original_sensor] = df
            logger.debug(f"센서 {original_sensor}: {len(df)}개 데이터 포인트 로드")

    # 시각화 생성
    logger.info("시각화 생성 중...")
    fig = create_interactive_plot(result)

    return fig, result


def create_interactive_plot(result: Dict[str, pd.DataFrame]) -> go.Figure:
    """예측 결과로부터 대화형 Plotly 차트를 생성합니다."""
    logger = logging.getLogger(__name__)
    logger.info("Plotly 시각화 준비 중...")

    # 데이터 전처리: 월별 정보 추가
    for sensor_name, df in result.items():
        df["year_month"] = df["time"].dt.to_period("M")
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

    # 전체 시간 범위 및 센서 목록 추출
    all_months = set()
    sensor_names = list(result.keys())

    for df in result.values():
        all_months.update(df["year_month"].unique())

    sorted_months = sorted(all_months)
    logger.info(f"전체 기간: {sorted_months[0]} ~ {sorted_months[-1]}")

    # 초기 데이터 설정 (첫 번째 센서, 첫 번째 월)
    initial_sensor = sensor_names[0]
    initial_month = sorted_months[0]
    initial_data = result[initial_sensor][
        result[initial_sensor]["year_month"] == initial_month
    ]

    # Plotly Figure 생성
    fig = go.Figure()

    # 초기 실제값과 예측값 라인 추가
    fig.add_trace(
        go.Scatter(
            x=initial_data["time"],
            y=initial_data["target"],
            name="실제값",
            line=dict(color="blue", width=2),
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=initial_data["time"],
            y=initial_data["prediction"],
            name="예측값",
            line=dict(color="red", width=2, dash="dash"),
            mode="lines",
        )
    )

    # 센서별 드롭다운 메뉴 생성
    dropdown_buttons = []
    for sensor in sensor_names:
        # 현재 선택된 월에 해당하는 센서 데이터
        sensor_month_data = result[sensor][
            result[sensor]["year_month"] == initial_month
        ]

        dropdown_buttons.append(
            {
                "label": f"센서 {sensor}",
                "method": "update",
                "args": [
                    {
                        "x": [sensor_month_data["time"], sensor_month_data["time"]],
                        "y": [
                            sensor_month_data["target"],
                            sensor_month_data["prediction"],
                        ],
                    },
                    {"title": f"센서 {sensor} - {initial_month} 교통량 예측"},
                ],
            }
        )

    # 월별 슬라이더 생성
    slider_steps = []
    for i, month in enumerate(sorted_months):
        # 현재 선택된 센서에 해당하는 월 데이터
        month_data = result[initial_sensor][
            result[initial_sensor]["year_month"] == month
        ]

        step = {
            "args": [
                {
                    "x": [month_data["time"], month_data["time"]],
                    "y": [month_data["target"], month_data["prediction"]],
                },
                {"title": f"센서 {initial_sensor} - {month} 교통량 예측"},
            ],
            "label": str(month),
            "method": "update",
        }
        slider_steps.append(step)

    # 레이아웃 설정
    fig.update_layout(
        title=f"센서 {initial_sensor} - {initial_month} 교통량 예측",
        xaxis_title="시간",
        yaxis_title="교통량",
        hovermode="x unified",
        # 드롭다운 메뉴 설정
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "lightgray",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        ],
        # 슬라이더 설정
        sliders=[
            {
                "steps": slider_steps,
                "currentvalue": {"prefix": "월: ", "visible": True, "xanchor": "right"},
                "pad": {"t": 50},
                "len": 0.8,
                "x": 0.1,
                "y": 0.02,
            }
        ],
        # 여백 설정
        margin=dict(t=150, b=100),
        height=600,
        # 범례 설정
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
