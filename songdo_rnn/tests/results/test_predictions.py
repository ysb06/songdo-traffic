from typing import Dict, List
import pytest
from scipy import stats
from songdo_rnn.plot import plot_loss
from songdo_rnn.utils import load_results_metrics

from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd


@pytest.fixture
def results_path():
    return "./output/predictions_refine"


def test_comparing_results(results_path: str):
    plot_loss(results_path)


def test_two_group_comparison(results_path: str):
    mae_results, rmse_results, smape_results, _ = load_results_metrics(results_path)

    print("\n=== MAE 결과 ===")
    two_group_comparison(mae_results)

    print("\n=== RMSE 결과 ===")
    two_group_comparison(rmse_results)

    print("\n=== SMAPE 결과 ===")
    two_group_comparison(smape_results)


def two_group_comparison(example_metrics: Dict[str, List[float]]):
    """
    metrics 딕셔너리에 여러 그룹이 있을 때,
    두 그룹씩만 꺼내서 t-검정을 수행하고 결과를 출력하는 예시.
    """
    groups = sorted(example_metrics.keys())  # 그룹 이름들을 정렬 (옵션)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_i = groups[i]
            group_j = groups[j]

            data_i = example_metrics[group_i]
            data_j = example_metrics[group_j]

            # 2집단 t-검정(독립표본)
            tstat, pvalue = stats.ttest_ind(data_i, data_j, equal_var=False)

            print(f"\n===== {group_i} vs {group_j} =====")
            print(f"  - mean({group_i}) = {pd.Series(data_i).mean():.4f}")
            print(f"  - mean({group_j}) = {pd.Series(data_j).mean():.4f}")
            print(f"  - t-stat = {tstat:.4f}, p-value = {pvalue:.4f}")
            if pvalue < 0.05:
                print("  -> 차이가 유의미함 (p<0.05)")
            else:
                print("  -> 통계적으로 유의한 차이 없음")


# def test_anova_results(results_path: str):
#     mae_results, rmse_results, smape_results, _ = load_results_metrics(results_path)

#     print("\n=== MAE 결과 ===")
#     anova_test(mae_results)

#     print("\n=== RMSE 결과 ===")
#     anova_test(rmse_results)

#     print("\n=== SMAPE 결과 ===")
#     anova_test(smape_results)


def anova_test(metrics: Dict[str, List[float]]):
    rows = []
    for group, values in metrics.items():
        for value in values:
            rows.append({"group": group, "value": value})
    df_metric = pd.DataFrame(rows)

    model = ols("value ~ C(group)", data=df_metric).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print(f"\n=== ANOVA 결과 ===")
    print(anova_table)

    if df_metric["group"].nunique() > 1:
        tukey_result = pairwise_tukeyhsd(
            endog=df_metric["value"],
            groups=df_metric["group"],
            alpha=0.05,
        )
        print(f"\n--- Tukey HSD 결과 ---")
        print(tukey_result)
        print("-----------------------------------")
    else:
        print(f"그룹이 1개뿐이어서 Tukey HSD 불가")
