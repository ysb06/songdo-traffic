import logging
from pprint import pprint

from .experiments.best_interpolation_search import do_experiment
from .experiments.prediction_evaluation import do_evaluation, plot_metrics
from .preprocessing.outlier import (
    process_outlier,
    remove_base_outliers_from_file,
    remove_outliers,
    get_outlier_removed_data_list,
)
from .preprocessing.missing import process_missing, interpolate_missing
from .preprocessing.sync import process_sync
from .plot import plot_loss


logger = logging.getLogger(__name__)

# # ----- 이상치 처리 ----- 
remove_base_outliers_from_file()

# 주의: 폴더가 비어있는 상태에서 Base만 생성된 상태여야 함
outlier_base_processed = get_outlier_removed_data_list()
remove_outliers(outlier_base_processed[0].data)
# outlier_processed = get_outlier_removed_data_list()

# # logger.info("Outlier Processed Data List")
# pprint([data.path for data in outlier_processed])

# # # ----- 결측치 처리 ----- 
# interpolate_missing(outlier_processed)

# # # ----- 공통 센서만 있는 데이터로 정제 ----- 
# process_sync()

# # # ----- RNN 학습 ----- 
# do_experiment(skip_preprocessing=True)
# plot_loss()
# do_evaluation()
# plot_metrics()

# Legacy Code
# process_outlier()
# process_missing()