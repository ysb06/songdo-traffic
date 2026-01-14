import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
import holidays
from .base import OutlierProcessor


class HMHZscoreProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0, years=None) -> None:
        super().__init__()
        self.threshold = threshold

    def _is_holiday(self, date: pd.Timestamp) -> bool:
        # 주말이거나 공휴일인 경우 True 반환
        korean_holidays = holidays.KR(years=date.year)
        return date.weekday() >= 5 or date in korean_holidays

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        def zscore_within_group(x: pd.Series) -> pd.Series:
            return (x - x.mean()) / x.std()

        # 휴일 여부 확인
        is_holiday = pd.Series(df.index.map(self._is_holiday), index=df.index)

        # 월, 시간, 휴일 여부로 그룹화
        grouped: DataFrameGroupBy = df.groupby(
            [df.index.month, df.index.hour, is_holiday]
        )
        zscored: pd.DataFrame = grouped.transform(zscore_within_group)

        if zscored.isna().all().all():
            self.successed_list.extend(df.columns)
        else:
            self.failed_list.extend(df.columns)

        df_clean = df.mask(zscored.abs() > self.threshold)
        return df_clean
