"""
Traffic Analysis Pipeline
클래스 기반 구조로 리팩토링된 교통 분석 파이프라인
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import pandas as pd
import plotly.graph_objects as go
import torch
from metr.datasets.rnn.datamodule import MultiSensorTrafficDataModule

from . import MODEL_OUTPUT_DIR
from .error_analysis import analyze_dataset_errors, save_error_analysis_results
from .prediction_comparison import analyze_predictions, analyze_predictions_from_file
from .utils import load_or_train_model, load_config

logger = logging.getLogger(__name__)


class TrafficAnalysisPipeline:
    """교통 예측 분석을 위한 메인 파이프라인 클래스"""

    def __init__(self, config_path: str = None):
        """
        파이프라인 초기화

        Args:
            config_path: 설정 파일 경로 (기본값: config.yaml)
        """
        # 설정 로드
        self.config: Dict[str, Dict[str, Any]] = load_config(config_path)
        self.analysis_config = self.config["analysis"]

        # 경로 설정
        self.dataset_path = self.analysis_config["dataset_path"]
        self.output_dir = MODEL_OUTPUT_DIR / "analysis_results"

        # 초기화할 속성들
        self.model = None
        self.data_module = None
        self.datasets = {}
        self.results: Dict[str, Tuple[go.Figure, Dict[str, pd.DataFrame], go.Figure, List, List]] = {}

        logger.info("TrafficAnalysisPipeline 초기화 완료")

    def setup_model_and_data(self):
        """모델과 데이터 모듈을 설정합니다."""
        logger.info("모델 로딩 시작...")
        self.model = load_or_train_model()
        self.model.eval()

        # 모델 디바이스 확인
        device = next(self.model.parameters()).device
        logger.info(f"모델 디바이스: {device}")

        # 데이터 모듈 초기화
        logger.info(f"데이터 로딩 시작: {self.dataset_path}")
        self.data_module = MultiSensorTrafficDataModule(
            dataset_path=self.dataset_path,
            shuffle_training=self.analysis_config.get("shuffle_training", False),
            scale_method=self.analysis_config.get("scale_method"),
        )
        self.data_module.setup()

        # 데이터로더 생성
        self.datasets = {
            "training": self.data_module.train_dataloader(),
            "validation": self.data_module.val_dataloader(),
            "test": self.data_module.test_dataloader(),
        }

        logger.info("모델 및 데이터 설정 완료")

    def analyze_single_dataset(
        self, dataset_name: str, dataloader
    ) -> Tuple[go.Figure, Dict[str, pd.DataFrame], go.Figure, List, List]:
        """단일 데이터셋에 대한 분석을 수행합니다.

        Args:
            dataset_name: 데이터셋 이름
            dataloader: 데이터로더

        Returns:
            분석 결과 튜플 (예측 그래프, 결과 데이터, 에러 그래프, 센서 메트릭, 상위 에러)
        """
        logger.info(f"{dataset_name.upper()} 데이터 분석 시작...")

        # 저장 경로 설정
        save_path = f"./output/analysis/model/rnn/traffic_prediction_{dataset_name}"
        h5_path = Path(save_path).with_suffix(".h5")
        pkl_path = Path(save_path).with_suffix(".pkl")

        # 저장된 결과가 있는지 확인
        if (
            h5_path.exists()
            and pkl_path.exists()
            and self.analysis_config.get("save_predictions", True)
        ):
            logger.info(f"저장된 결과 파일 발견: {save_path}")
            logger.info("예측을 건너뛰고 저장된 결과 로드...")
            fig, result = analyze_predictions_from_file(save_path)
        else:
            logger.info("새로운 예측 수행...")
            device = next(self.model.parameters()).device
            fig, result = analyze_predictions(
                dataloader,
                self.model,
                device,
                save_path=(
                    save_path
                    if self.analysis_config.get("save_predictions", True)
                    else None
                ),
            )

        # 에러 분석 수행
        logger.info(f"{dataset_name} 데이터셋 에러 분석 수행...")
        sensor_metrics, top_errors, error_fig = analyze_dataset_errors(
            result, dataset_name
        )

        logger.info(f"{dataset_name} 분석 완료")
        return fig, result, error_fig, sensor_metrics, top_errors

    def analyze_all_datasets(self):
        """모든 데이터셋에 대한 분석을 수행합니다."""
        logger.info("전체 데이터셋 분석 시작...")

        for dataset_name, dataloader in self.datasets.items():
            analysis_result = self.analyze_single_dataset(dataset_name, dataloader)
            self.results[dataset_name] = analysis_result

        logger.info("전체 데이터셋 분석 완료")

    def save_results(self):
        """분석 결과를 저장합니다."""
        logger.info("결과 저장 시작...")

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plotly_config: Dict[str, Any] = self.analysis_config.get("plotly", {})

        # 각 데이터셋별로 결과 저장
        for dataset_name, (
            fig,
            result,
            error_fig,
            sensor_metrics,
            top_errors,
        ) in self.results.items():
            # 예측 비교 그래프 저장
            html_path = self.output_dir / f"traffic_prediction_{dataset_name}.html"
            fig.write_html(
                str(html_path),
                include_plotlyjs=plotly_config.get("include_plotlyjs", "cdn"),
                config={
                    "displayModeBar": plotly_config.get("display_mode_bar", True),
                    "responsive": plotly_config.get("responsive", True),
                },
            )
            logger.info(f"{dataset_name} 예측 그래프 저장: {html_path}")

            # 에러 분석 그래프 저장
            error_html_path = self.output_dir / f"error_analysis_{dataset_name}.html"
            error_fig.write_html(
                str(error_html_path),
                include_plotlyjs=plotly_config.get("include_plotlyjs", "cdn"),
                config={
                    "displayModeBar": plotly_config.get("display_mode_bar", True),
                    "responsive": plotly_config.get("responsive", True),
                },
            )
            logger.info(f"{dataset_name} 에러 분석 그래프 저장: {error_html_path}")

            # CSV 결과 저장
            save_error_analysis_results(
                sensor_metrics, top_errors, dataset_name, str(self.output_dir)
            )

        logger.info(f"모든 결과 저장 완료: {self.output_dir}")

    def print_summary(self):
        """분석 결과 요약을 출력합니다."""
        logger.info("=== 분석 결과 요약 ===")
        logger.info("저장된 파일들:")

        for dataset_name in self.datasets.keys():
            logger.info(
                f"- traffic_prediction_{dataset_name}.html: {dataset_name} 데이터 예측 비교 그래프"
            )
            logger.info(
                f"- error_analysis_{dataset_name}.html: {dataset_name} 데이터 에러 분석 그래프"
            )
            logger.info(
                f"- error_metrics_{dataset_name}.csv: {dataset_name} 데이터 센서별 메트릭"
            )
            logger.info(
                f"- top_errors_{dataset_name}.csv: {dataset_name} 데이터 Top 10 에러 케이스"
            )

        logger.info("📊 예측 비교 그래프에서:")
        logger.info("- 상단 드롭다운으로 센서를 선택할 수 있습니다")
        logger.info("- 하단 슬라이더로 월을 선택할 수 있습니다")
        logger.info("- 범례를 클릭하여 라인을 숨기거나 표시할 수 있습니다")

        logger.info("📈 에러 분석 그래프에서:")
        logger.info("- 센서별 MAE 분포 히스토그램")
        logger.info("- RMSE vs MAE 센서별 성능 산점도")
        logger.info("- 가장 큰 에러 Top 10 바차트")
        logger.info("- MAPE vs R² 상관관계 분석")

    def run_complete_analysis(self):
        """전체 분석 파이프라인을 실행합니다."""
        logger.info("=== 교통 분석 파이프라인 시작 ===")

        try:
            # 1. 모델과 데이터 설정
            self.setup_model_and_data()

            # 2. 모든 데이터셋 분석
            self.analyze_all_datasets()

            # 3. 결과 저장
            self.save_results()

            # 4. 결과 요약 출력
            self.print_summary()

            logger.info("=== 교통 분석 파이프라인 완료 ===")

        except Exception as e:
            logger.error(f"분석 파이프라인 실행 중 오류 발생: {e}")
            raise
