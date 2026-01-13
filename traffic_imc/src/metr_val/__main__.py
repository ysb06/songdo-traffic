from .stgcn import main

SMALL_DATASET_DIR = "./data/selected_small_v1"
MAIN_DATASET_DIR = "../datasets/metr-imc/subsets/v2"

main(MAIN_DATASET_DIR)