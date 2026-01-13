import logging

from metr.pipeline import generate_raw_dataset, generate_subset_dataset


logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

# generate_raw_dataset()
generate_subset_dataset(
    # target_nodelinks_path="../datasets/metr-imc/subsets/v1/nodelink/imc_link.shp",
    target_data_start="2020-01-26 00:00:00",
)
