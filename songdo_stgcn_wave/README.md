# songdo_stgcn_wave

이 프로젝트는 STGCN WAVE 모델을 기반으로 인천시 교통량 예측 모델을 생성하는 것이 목표로 이를 위한 다양한 코드를 포함하고 있습니다.

This project aims to create a traffic volume prediction model for Incheon City based on the STGCN WAVE model and includes various codes for this purpose.

## Get Started for training model

### Installation

1. Move to workspace root folder

2. Install desired PyTorch and DGL

    Example for DGL 2.4.0 with CUDA 12.1 with Linux:
    ```
    pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    ```
    Note 1: DGL 2.4.0 is only compatible with PyTorch 2.4.0 and 2.4.1
    
    Note 2: When you install desired DGL, you must specify the version like dgl==2.4.0 to upgrade from basically installed DGL.

3. Install "songdo_metr" and this project
    ```
    pip install ./songdo_metr
    pip install ./songdo_stgcn_wave
    ```

자세한 설치는 다음 사이트를 참고하세요:
- https://pytorch.org/get-started/locally/
- https://www.dgl.ai/pages/start.html

### Running the Code

```bash
python -m songdo_stgcn_trainer --config imc-base
```

코드를 실행하면 config 매개변수에 지정된 이름의 configs 폴더 내 yaml파일을 읽습니다. config 매개변수가 없을 경우 무조건 base.yaml을 읽습니다. 이 config는 원래 이 STGCN 모델 코드의 소스에서 지정한 하이퍼파라미터와 같은 값을 가지고 있습니다.

추가로 하이퍼파라미터를 임의로 매개변수로 지정하면 읽은 config 파일의 하이퍼파라미터 값을 강제로 재설정합니다. 예를 들어, epoch를 yaml에서 지정된 50이 아닌 100을 설정하고 싶을 경우 다음과 같이 실행하세요.

```bash
python -m songdo_stgcn_wave --config imc-base --epochs 100
```


## References

1. https://github.com/dmlc/dgl/tree/master/examples/pytorch/stgcn_wave