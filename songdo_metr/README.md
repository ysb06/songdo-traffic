# songdo_metr

이 프로젝트는 인천광역시 교통량 데이터 및 한국 도로망 데이터를 수집하고 예측 모델 학습을 위한 데이터셋을 생성하는 것을 목표로 합니다. 이렇게 생성된 데이터셋은 인천광역시 내 도로의 공간 구조를 그래프 형태로 나타낸 인접 행렬과 시간에 따른 교통량 변화를 포함한 시계열 데이터를 모두 포함합니다. 이 데이터셋은 교통량 에측과 관련하여 많이 알려진 METR-LA 데이터셋과 호환 가능한 데이터셋을 생성하도록 되어 있습니다. 추가로, 이 프로젝트에는 결측치 처리와 같은 데이터 보정을 위한 유틸리티도 포함합니다.

This project aims to collect traffic data and road network data for Incheon Metropolitan City and generate datasets for training predictive models. The generated datasets include both adjacency matrices representing the spatial structure of roads in Incheon and time series data reflecting traffic volume changes over time. These datasets are designed to be compatible with the well-known METR-LA dataset used in traffic volume prediction. Additionally, this project includes utilities for data correction, such as handling missing values.

## Get Started

This project provides a script that automates the process of downloading data from the original sources, including the Korean Standard Node-Link and Incheon City traffic data, and converting it into a usable dataset.

### Installing Requirements
#### Using PDM

```bash
pdm install
```

#### Not Using PDM

```bash
pip install -r requirements.txt
```

### Running the Code

1. Set Environment Variables

To access Incheon City traffic data, you need to sign up for data.go.kr and register your API Key as an environment variable. Since this project uses a src layout, set the environment variables as shown below:

```bash
export PYTHONPATH="./src"
export PDP_KEY = "[API Key for data.go.kr]"
```

Alternatively, you can create a .env file in the project folder with the following content:

```
PYTHONPATH="./src"
PDP_KEY = "[API Key for data.go.kr]"
```

2. Run the Code

```bash
python -m metr.dataset
```