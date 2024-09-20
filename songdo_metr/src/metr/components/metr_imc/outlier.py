import pandas as pd


class OutlierProcessor:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class ZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean()
        std = df.std()
        z_scores = (df - mean) / std
        outliers = z_scores > self.threshold
        df_clean = df.mask(outliers)

        return df_clean
    # Todo: Complete Components