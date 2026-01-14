from .rnn import main

SMALL_DATASET_DIR = "./data/selected_small_v1"
MAIN_DATASET_DIR = "../datasets/metr-imc/subsets/v2"
MAIN_DATASET_PATH = f"{MAIN_DATASET_DIR}/metr-imc.h5"

main(MAIN_DATASET_PATH)

# Todo: 학습-검증-테스트 데이터셋을 7:2:1 비율로 나누기