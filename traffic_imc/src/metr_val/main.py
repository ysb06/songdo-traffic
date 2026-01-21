from .mlcaformer import main as mlcaformer_main
from .dcrnn import main as dcrnn_main
from .stgcn import main as stgcn_main
from .agcrn import main as agcrn_main
from .rnn import main as lstm_main
from metr.utils import PathConfig

def sequential_main():
    # path_config = PathConfig.from_yaml("../config_knn.yaml")
    # mlcaformer_main("KNN", path_config, code=0)
    data_list = [
        ("KNN", "../config_knn.yaml"),
        ("MICE", "../config_mice.yaml"),
        ("BGCP", "../config_bgcp.yaml"),
        ("TRMF", "../config_trmf.yaml"),
        ("BRITS", "../config_brits.yaml"),
    ]

    model_training_list = [
        stgcn_main,
        dcrnn_main,
        agcrn_main,
        mlcaformer_main,
        lstm_main,
    ]
    for data_name, config_path in data_list:
        for model_main in model_training_list:
            path_config = PathConfig.from_yaml(config_path)
            model_main(data_name, path_config, code=2)