from collections import defaultdict
from typing import Dict, List, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .utils import symmetric_mean_absolute_percentage_error


def do_simple_replacement_test(
    test_set: List[Tuple[pd.DataFrame, str]],
    true_df: pd.DataFrame,
    diff_mask: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_results = {}
    mae_sensor_results = defaultdict(list)
    rmse_sensor_results = defaultdict(list)
    smape_sensor_results = defaultdict(list)
    sensor_results_index: List[str] = []
    for target_df, target_name in test_set:
        test_target_df = target_df.loc[true_df.index]
        target_mask = diff_mask.loc[true_df.index]
        result = calculate_metrics(
            y_true=true_df, y_pred=test_target_df, target_mask=target_mask
        )
        total_metrics: Dict[str, Union[int, float]] = result["total"]
        each_sensor_metrics: Dict[str, Dict[str, Union[int, float]]] = result["sensor"]
        total_results[target_name] = {
            "mae": total_metrics["mae"],
            "rmse": total_metrics["rmse"],
            "smape": total_metrics["smape"],
            "count": total_metrics["count"],
        }
        sensor_results_index.append(target_name)
        for sensor_name, metrics in each_sensor_metrics.items():
            mae_sensor_results[sensor_name].append(metrics["mae"])
            rmse_sensor_results[sensor_name].append(metrics["rmse"])
            smape_sensor_results[sensor_name].append(metrics["smape"])

    # 센서별 메트릭을 데이터프레임으로 변환
    total_results_df = pd.DataFrame(total_results).T
    mae_sensor_results_df = pd.DataFrame(mae_sensor_results, index=sensor_results_index)
    rmse_sensor_results_df = pd.DataFrame(
        rmse_sensor_results, index=sensor_results_index
    )
    smape_sensor_results_df = pd.DataFrame(
        smape_sensor_results, index=sensor_results_index
    )

    return (
        total_results_df,
        mae_sensor_results_df,
        rmse_sensor_results_df,
        smape_sensor_results_df,
    )


def calculate_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    target_mask: pd.Series,
    skip_na: bool = True,
) -> Dict[str, Union[float, Dict[str, Dict[str, float]]]]:
    """
    두 데이터프레임 간의 MAE와 RMSE를 계산

    Parameters:
    -----------
    y_true : pd.DataFrame - 참값 데이터프레임 (test_df)
    y_pred : pd.DataFrame - 예측값 데이터프레임
    skip_na : bool - True면 NaN 값은 계산에서 제외

    Returns:
    --------
    dict - 전체 및 센서별 메트릭 결과
    """
    # Shape 확인
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"데이터프레임 shape이 일치하지 않습니다: {y_true.shape} vs {y_pred.shape}"
        )

    # 전체 데이터에 대한 메트릭
    if skip_na:
        # NaN이 있는 위치는 계산에서 제외
        mask = ~(y_true.isna() | y_pred.isna())
        true_values = y_true.values[mask.values]
        pred_values = y_pred.values[mask.values]
    else:
        true_values = y_true.values.flatten()
        pred_values = y_pred.values.flatten()

    global_mae = mean_absolute_error(true_values, pred_values)
    global_rmse = root_mean_squared_error(true_values, pred_values)
    global_smape = symmetric_mean_absolute_percentage_error(true_values, pred_values)

    # 센서별 메트릭 계산
    per_sensor_metrics = {}
    for col in y_true.columns:
        if skip_na:
            mask = ~(y_true[col].isna() | y_pred[col].isna())
            col_true = y_true.loc[mask, col]
            col_pred = y_pred.loc[mask, col]
        else:
            col_true = y_true[col].dropna()
            col_pred = y_pred[col].dropna()

        if len(col_true) > 0:
            mae = mean_absolute_error(col_true, col_pred)
            rmse = root_mean_squared_error(col_true, col_pred)
            smape = symmetric_mean_absolute_percentage_error(col_true, col_pred)
            per_sensor_metrics[col] = {
                "mae": mae,
                "rmse": rmse,
                "smape": smape,
                "count": len(col_true),
            }

    return {
        "total": {
            "mae": global_mae,
            "rmse": global_rmse,
            "smape": global_smape,
            "count": len(true_values),
        },
        "sensor": per_sensor_metrics,
    }


if __name__ == "__main__":
    test_true_df = pd.read_hdf("./output/test_true.h5")
