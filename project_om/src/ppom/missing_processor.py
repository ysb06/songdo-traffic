import glob
import logging
import os
from typing import List, Optional, Tuple

from metr.components import TrafficData
from metr.components.metr_imc.interpolation import (
    Interpolator,
    LinearInterpolator,
    SplineLinearInterpolator,
    TimeMeanFillInterpolator,
    ShiftFillInterpolator,
    MonthlyMeanFillInterpolator,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTLIER_PROCESSED_DIR = "./output/outlier_processed"
OUTPUT_DIR = "./output/missing_processed"

logger = logging.getLogger(__name__)


def interpolate(
    df_list: List[Tuple[pd.DataFrame, str]],
    interpolators: List[Interpolator],
    output_dir: Optional[str] = None,
) -> List[Tuple[pd.DataFrame, str]]:
    outputs = []

    total_length = len(df_list) * len(interpolators)
    logger.info(f"Results will be saved to {output_dir}" if output_dir else "Result saving disabled")
    with tqdm(total=total_length, desc="Interpolating") as pbar:
        for idx_1, raw in enumerate(df_list):
            data, data_basename = raw

            for idx_2, interpolator in enumerate(interpolators):
                name = f"{data_basename}-{interpolator.name}"
                pbar.set_description(f"Interpolating {name}...")

                interpolated_data = interpolator.interpolate(data)
                outputs.append((interpolated_data, name))

                if output_dir is not None:
                    pbar.set_description(f"Saving {name}...")
                    filepath = os.path.join(output_dir, f"{name}.h5")
                    interpolated_data.to_hdf(filepath, key="data")
                    pbar.update(1)

    return outputs
