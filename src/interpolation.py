import numpy as np
import pandas as pd

from songdo_traffic_core.dataset.interpolator import (
    IterativeRandomForestInterpolator, LinearInterpolator, SplineInterpolator)
from songdo_traffic_core.dataset.metr_imc.generator import \
    MetrImcSubsetGenerator


def extend_nans_around_zeros(series: pd.Series) -> pd.Series:
    series = series.copy()
    nan_indices = series[series.isna()].index
    
    for idx in nan_indices:
        idx_pos = series.index.get_loc(idx)
        
        i = idx_pos - 1
        while i >= 0 and series.iat[i] == 0:
            series.iat[i] = np.nan
            i -= 1
            
        i = idx_pos + 1
        while i < len(series) and series.iat[i] == 0:
            series.iat[i] = np.nan
            i += 1
    
    return series

df_imc: pd.DataFrame = pd.read_hdf("./datasets/metr-imc/metr-imc.h5")

no_missing_columns = df_imc.columns[df_imc.isnull().sum() == 0].to_list()
less_500_missing_columns = df_imc.columns[df_imc.isnull().sum() < 500].to_list()
less_750_missing_columns = df_imc.columns[df_imc.isnull().sum() < 750].to_list()

generator = MetrImcSubsetGenerator(
    nodelink_dir="./datasets/metr-imc/nodelink",
    imcrts_dir="./datasets/metr-imc/imcrts",
    metr_imc_dir="./datasets/metr-imc/",
)

# interpolator = IterativeRandomForestInterpolator(verbose=1)
interpolator = SplineInterpolator()

generator.generate_subset(
    less_500_missing_columns, "./datasets/metr-imc-296-no-interpolation",
)

# 결측치 주변의 0값을 모두 결측치로 처리
df = generator.metr_imc_df
df = df.sort_index()
generator.metr_imc_df = df.apply(extend_nans_around_zeros)

generator.generate_subset(
    less_500_missing_columns, "./datasets/metr-imc-296-interpolation", interpolator,
)


