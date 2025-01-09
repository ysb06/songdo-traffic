import logging

from .experiments.best_interpolation_search import do_experiment


logger = logging.getLogger(__name__)

# process_outlier()
# process_missing()

do_experiment()
