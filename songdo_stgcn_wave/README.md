# songdo_stgcn_wave

## Get Started

### Installing Requirements

#### Using PDM

```bash
pdm install
```

#### Not Using PDM

```bash
pip install -r requirements.txt
```

### Setting Environment Variables

```bash
export PYTHONPATH="./src"
```

Alternatively, you can create a .env file in the project folder with the following content:

```
PYTHONPATH="./src"
```

### Running the Code

```bash
python -m songdo_stgcn_wave --config imc-base
```

코드를 실행하면 config 매개변수에 지정된 이름의 configs 폴더 내 yaml파일을 읽습니다. config 매개변수가 없을 경우 무조건 base.yaml을 읽습니다. 이 config는 원래 이 STGCN 모델 코드의 소스에서 지정한 하이퍼파라미터와 같은 값을 가지고 있습니다.

추가로 하이퍼파라미터를 임의로 매개변수로 지정하면 읽은 config 파일의 하이퍼파라미터 값을 강제로 재설정합니다. 예를 들어, epoch를 yaml에서 지정된 50이 아닌 100을 설정하고 싶을 경우 다음과 같이 실행하세요.

```bash
python -m songdo_stgcn_wave --config imc-base --epochs 100
```


## References

1. https://github.com/dmlc/dgl/tree/master/examples/pytorch/stgcn_wave