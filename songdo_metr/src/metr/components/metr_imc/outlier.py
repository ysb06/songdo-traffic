import pandas as pd


class OutlierProcessor:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

class ZscoreOutlierProcessor(OutlierProcessor):
    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    # Todo: Implement ZscoreOutlierProcessor
    # Todo: Complete Components