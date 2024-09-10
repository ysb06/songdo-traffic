# 인천시 교통량 데이터셋을 다루기 위한 모듈
import os

IMCRTS_DEFAULT_DIR = "../datasets/metr-imc/imcrts"
PDP_KEY = os.environ.get("PDP_KEY")

os.makedirs(IMCRTS_DEFAULT_DIR, exist_ok=True)
