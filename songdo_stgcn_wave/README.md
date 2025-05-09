# songdo_stgcn_wave

이 프로젝트는 STGCN WAVE 모델을 기반으로 인천시 교통량 예측 모델을 생성하는 것을 목표로 하며, 이를 위한 다양한 코드를 포함하고 있습니다.

This project aims to develop a traffic volume prediction model for Incheon City based on the STGCN WAVE model and includes various code for this purpose.

## Getting Started with Training the Model

### Installation

1. Move to the workspace root folder.

2. Install the desired versions of PyTorch and DGL.

    Example installation for DGL 2.4.0 with CUDA 12.1 on Linux:
    ```bash
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
    ```
    Note 1: DGL 2.4.0 is only compatible with PyTorch 2.4.0.

    Note 2: When installing DGL, you must specify the version (e.g., `dgl==2.4.0`) to upgrade from the default installed DGL.

    Note 3: PyTorch should be installed first, before installing DGL.

3. Install "songdo_metr" and this project:
    ```bash
    pip install ./songdo_metr
    pip install ./songdo_stgcn_wave
    ```

자세한 설치는 다음 사이트를 참고하세요:

For detailed installation instructions, please refer to the following sites:

- https://pytorch.org/get-started/locally/
- https://www.dgl.ai/pages/start.html

### Running the Code

```bash
python -m songdo_stgcn_trainer --config imc-base
```

코드를 실행하면 config 매개변수에 지정된 이름의 configs 폴더 내 YAML 파일을 읽습니다. config 매개변수가 없을 경우 기본적으로 base.yaml을 읽습니다. 이 설정 파일은 원래 STGCN 모델 코드의 소스에서 지정한 하이퍼파라미터와 동일한 값을 가지고 있습니다.

When you run the code, it reads the YAML file in the configs folder with the name specified in the config parameter. If the config parameter is not provided, it will read base.yaml by default. This config contains the same hyperparameter values as those specified in the original STGCN model source code.

추가로 하이퍼파라미터를 임의로 매개변수로 지정하면, 읽은 설정 파일의 하이퍼파라미터 값을 강제로 재설정합니다. 예를 들어, epochs를 YAML에서 지정된 50이 아닌 100으로 설정하고 싶을 경우 다음과 같이 실행하세요.

Additionally, if you manually specify hyperparameters as parameters, it will override the hyperparameter values in the config file that was read. For example, if you want to set the epochs to 100 instead of the 50 specified in the YAML, run the following command:

```bash
python -m songdo_stgcn_wave --config imc-base --epochs 100
```


## References

1. https://github.com/dmlc/dgl/tree/master/examples/pytorch/stgcn_wave