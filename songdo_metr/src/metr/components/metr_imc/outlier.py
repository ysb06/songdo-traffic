import numpy as np
import pandas as pd


class OutlierProcessor:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class ZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
    
    def __detect_outliers_zscore_df(self, df: pd.DataFrame, threshold=3) -> pd.DataFrame:
        # 시간대 별로 데이터를 그룹화하여 z-score 계산
        outliers = pd.DataFrame(index=df.index, columns=df.columns, dtype=bool)
        
        info = []
        for hour in range(24):
            hourly_data: pd.DataFrame = df[df.index.hour == hour]

            mean = hourly_data.stack().mean()  # 시간대별 모든 값의 평균
            std = hourly_data.stack().std()  # 시간대별 모든 값의 표준편차
            z_scores = (hourly_data - mean) / std

            outliers.loc[hourly_data.index] = np.abs(z_scores) > threshold
            info.append({
                "hour": hour,
                "mean": mean,
                "std": std,
                "threshold": threshold,
                "outlier_threshold": mean + threshold * std
            })

        outliers = outliers.convert_dtypes()
        dtypes = outliers.dtypes.unique()
        if len(dtypes) > 1:
            print("Warning: Different data types detected in the outlier dataframe.")
        return outliers, info

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        outliers, _ = self.__detect_outliers_zscore_df(df, self.threshold)
        df_clean = df.mask(outliers)

        return df_clean
    # Todo: Complete Components