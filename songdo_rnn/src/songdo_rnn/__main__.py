import logging
from pprint import pprint

from .experiments.best_interpolation_search import do_experiment
from .preprocessing.outlier import (
    process_outlier,
    remove_base_outliers,
    remove_outliers,
    get_outlier_removed_data_list,
)
from .preprocessing.missing import process_missing, interpolate_missing
from .preprocessing.sync import process_sync
from .plot import plot_loss


logger = logging.getLogger(__name__)

# ----- 이상치 처리 ----- 
# remove_base_outliers()
# outlier_base_processed = get_outlier_removed_data_list()
# remove_outliers(outlier_base_processed)
# outlier_processed = get_outlier_removed_data_list()

# logger.info("Outlier Processed Data List")
# pprint([data.path for data in outlier_processed])

# ----- 결측치 처리 ----- 
# interpolate_missing(outlier_processed)

# ----- 공통 센서만 있는 데이터로 정제 ----- 
# process_sync()

# ----- RNN 학습 ----- 
# do_experiment(skip_preprocessing=True)
plot_loss()

# Legacy Code
# process_outlier()
# process_missing()