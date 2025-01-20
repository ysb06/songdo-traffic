import logging

from .experiments.best_interpolation_search import do_experiment
from .preprocessing.outlier import process_outlier
from .preprocessing.missing import process_missing
from .plot import plot_loss

logger = logging.getLogger(__name__)



# do_experiment(skip_preprocessing=True)
plot_loss()
