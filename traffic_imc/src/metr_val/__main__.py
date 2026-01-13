from .rnn import main

SMALL_DATASET_PATH = "./data/selected_small_v1/metr-imc.h5"
MAIN_DATASET_PATH = "../datasets/metr-imc/subsets/v2/metr-imc.h5"

main(MAIN_DATASET_PATH)