import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import torch
from metr.datasets.rnn.datamodule import MultiSensorTrafficDataModule
from tqdm import tqdm

from . import MODEL_OUTPUT_DIR
from .error_analysis import analyze_dataset_errors, save_error_analysis_results
from .pipeline import TrafficAnalysisPipeline
from .prediction_comparison import analyze_predictions, analyze_predictions_from_file
from .utils import load_or_train_model

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

# 새로운 클래스 기반 파이프라인 사용
if __name__ == "__main__":
    pipeline = TrafficAnalysisPipeline()
    pipeline.run_complete_analysis()
else:
    # 기존 코드 (하위 호환성을 위해 유지)
    analysis_target_model = load_or_train_model()

    # 데이터 경로 설정
    dataset_path = "./data/selected_small_v1/metr-imc.h5"
    print(f"데이터 경로: {dataset_path}")

    # MultiSensorTrafficDataModule 초기화
    print("MultiSensorTrafficDataModule 초기화 중...")
    data_module = MultiSensorTrafficDataModule(
        dataset_path=dataset_path, shuffle_training=False, scale_method=None
    )
    data_module.setup()

    training_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # 모델을 평가 모드로 설정
    analysis_target_model.eval()

    # 모델의 디바이스 확인
    device = next(analysis_target_model.parameters()).device
    print(f"모델 디바이스: {device}")

    # 각 데이터셋에 대해 예측 및 분석 수행
    datasets = {
        "training": training_loader,
        "validation": validation_loader,
        "test": test_loader,
    }

    results: Dict[
        str, Tuple[go.Figure, Dict[str, pd.DataFrame], go.Figure, list, list]
    ] = {}

    for dataset_name, dataloader in datasets.items():
        print(f"\n{dataset_name.upper()} 데이터에 대한 분석 중...")

        # 저장 경로 설정
        save_path = f"./output/analysis/model/rnn/traffic_prediction_{dataset_name}"

        # 저장된 파일 존재 확인
        h5_path = Path(save_path).with_suffix(".h5")
        pkl_path = Path(save_path).with_suffix(".pkl")

        if h5_path.exists() and pkl_path.exists():
            print(f"저장된 결과 파일을 발견했습니다: {save_path}")
            print("예측을 건너뛰고 저장된 결과를 불러옵니다...")

            # 저장된 파일에서 결과 로드
            fig, result = analyze_predictions_from_file(save_path)
        else:
            print("저장된 결과 파일이 없습니다. 새로운 예측을 수행합니다...")

            # prediction_comparison.py의 analyze_predictions 함수 사용
            fig, result = analyze_predictions(
                dataloader,
                analysis_target_model,
                device,
                save_path=save_path,
            )

        # 결과 저장
        results[dataset_name] = (fig, result)

        # 에러 분석 수행
        print(f"{dataset_name} 데이터셋 에러 분석 수행 중...")
        sensor_metrics, top_errors, error_fig = analyze_dataset_errors(
            result, dataset_name
        )

        # 에러 분석 결과도 저장
        results[dataset_name] = (fig, result, error_fig, sensor_metrics, top_errors)

        print(f"{dataset_name} 분석 완료!")

    # 결과 저장
    print("\n결과를 저장하는 중...")

    # 출력 디렉토리 생성
    output_dir = MODEL_OUTPUT_DIR / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 데이터셋별로 HTML 파일 저장
    for dataset_name, (
        fig,
        result,
        error_fig,
        sensor_metrics,
        top_errors,
    ) in results.items():
        # 예측 비교 그래프 저장
        html_path = output_dir / f"traffic_prediction_{dataset_name}.html"
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "responsive": True},
        )
        print(f"{dataset_name} 인터랙티브 그래프 저장: {html_path}")

        # 에러 분석 그래프 저장
        error_html_path = output_dir / f"error_analysis_{dataset_name}.html"
        error_fig.write_html(
            str(error_html_path),
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "responsive": True},
        )
        print(f"{dataset_name} 에러 분석 그래프 저장: {error_html_path}")

        # 에러 분석 결과를 CSV로 저장
        save_error_analysis_results(
            sensor_metrics, top_errors, dataset_name, str(output_dir)
        )

    print(f"\n모든 결과가 저장되었습니다: {output_dir}")
    print("저장된 파일들:")
    for dataset_name in datasets.keys():
        print(
            f"- traffic_prediction_{dataset_name}.html: {dataset_name} 데이터 예측 비교 그래프"
        )
        print(
            f"- error_analysis_{dataset_name}.html: {dataset_name} 데이터 에러 분석 그래프"
        )
        print(
            f"- error_metrics_{dataset_name}.csv: {dataset_name} 데이터 센서별 메트릭"
        )
        print(
            f"- top_errors_{dataset_name}.csv: {dataset_name} 데이터 Top 10 에러 케이스"
        )

    print("\n📊 예측 비교 그래프에서:")
    print("- 상단 드롭다운으로 센서를 선택할 수 있습니다")
    print("- 하단 슬라이더로 월을 선택할 수 있습니다")
    print("- 범례를 클릭하여 라인을 숨기거나 표시할 수 있습니다")

    print("\n📈 에러 분석 그래프에서:")
    print("- 센서별 MAE 분포 히스토그램")
    print("- RMSE vs MAE 센서별 성능 산점도")
    print("- 가장 큰 에러 Top 10 바차트")
    print("- MAPE vs R² 상관관계 분석")
