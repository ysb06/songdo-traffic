# songdo-traffic

이 프로젝트는 교통량 데이터를 기반으로 도심의 교통량에 대한 예측하는 것을 목표로 합니다. 이 프로젝트에는 데이터 수집 및 변환, 시각화 그리고 예측 모델 학습 등 이 프로젝트의 목표를 수행하기 위한 다양한 하위 프로젝트들이 포함되어 있습니다.

프로젝트 이름에는 인천시의 일부인 Songdo가 포함되어 있지만, 교통량 예측 대상은 Songdo 지역에만 제한되어 있지 않습니다. 현재, 프로젝트에는 인천시 전체의 데이터를 다운로드하고 변환하는 과정이 포함되어 있으며, 나아가 다른 지역의 도시 내 교통 데이터에도 대응 가능한 모델을 만드는 것이 목표입니다.

This project aims to predict urban traffic volume based on traffic data. It includes various sub-projects such as data collection and transformation, visualization, and training predictive models to achieve this goal.

Although the project name includes “Songdo,” which is part of Incheon, the traffic prediction is not limited to the Songdo area. Currently, the project involves downloading and transforming data for the entire Incheon city, with the goal of eventually creating a model that can be applied to urban traffic data from other regions as well.

## 주요 하위 프로젝트

이 단락에는 각 하위 프로젝트의 요약만 제공합니다. 코드 실행과 같은 세부적인 내용은 각 하위 프로젝트 폴더의 README에서 확인할 수 있습니다.

This section provides a summary of each sub-project. For detailed information such as code execution, please refer to the README file within each sub-project folder.

### songdo_metr

[document](./songdo_metr/README.md)

이 프로젝트는 인천광역시 교통량 데이터 및 한국 도로망 데이터를 수집하고 예측 모델 학습을 위한 데이터셋을 생성하는 것을 목표로 합니다. 이렇게 생성된 데이터셋은 인천광역시 내 도로의 공간 구조를 그래프 형태로 나타낸 인접 행렬과 시간에 따른 교통량 변화를 포함한 시계열 데이터를 모두 포함합니다. 이 데이터셋은 교통량 에측과 관련하여 많이 알려진 METR-LA 데이터셋과 호환 가능한 데이터셋을 생성하도록 되어 있습니다. 추가로, 이 프로젝트에는 결측치 처리와 같은 데이터 보정을 위한 유틸리티도 포함합니다.

This project aims to collect traffic data and road network data for Incheon Metropolitan City and generate datasets for training predictive models. The generated datasets include both adjacency matrices representing the spatial structure of roads in Incheon and time series data reflecting traffic volume changes over time. These datasets are designed to be compatible with the well-known METR-LA dataset used in traffic volume prediction. Additionally, this project includes utilities for data correction, such as handling missing values.

### songdo_qgis

이 프로젝트는 교통량 예측 모델 생성에 사용되는 데이터를 QGIS를 통해 시각화하는 것을 목표로 합니다. QGIS 플러그인 형태로 설치되어 실행되는 이 프로젝트를 통해 데이터셋의 특징을 파악하고 및 생성된 모델의 예측 결과를 시각적으로 확인할 수 있습니다.

This project aims to visualize the data used in traffic prediction model creation through QGIS. Installed and run as a QGIS plugin, this project allows users to understand dataset characteristics and visually inspect the prediction results of generated models.

### songdo_stgcn_wave

[document](./songdo_stgcn_wave/README.md)

이 프로젝트는 STGCN WAVE 모델을 기반으로 인천시 교통량 예측 모델을 생성하는 것이 목표로 이를 위한 다양한 코드를 포함하고 있습니다.

This project aims to create a traffic volume prediction model for Incheon City based on the STGCN WAVE model and includes various codes for this purpose.

### notebooks

Jupyter Notebook을 통해 하위 프로젝트를 활용하여 진행한 연구내용들이 포함되어 있습니다.

This directory includes research conducted using the sub-projects, utilizing Jupyter Notebooks.


## License

이 프로젝트에는 여러 개의 하위 프로젝트로 이루어져 있으며 서로 다른 개별의 라이선스를 가지고 있으므로 이 프로젝트 활용 시 각각의 내용을 반드시 확인 및 검토하십시오. 기본적으로 각 하위 프로젝트의 라이선스 독립을 위해 하위 프로젝트 간 직접적인 참조는 현재도 그리고 앞으로도 하지 않습니다. 다만, 각 하위 프로젝트의 결과물을 참조하는 형태로 간접적으로 연결될 수 있습니다.

This project consists of several sub-projects, each with its own license. Please be sure to review and confirm the license terms before using any part of this project. To maintain the independence of each sub-project’s license, there are no direct references between sub-projects, now or in the future. However, indirect connections may occur by referencing the outputs of each sub-project.
