from .mlcaformer import main
from metr.utils import PathConfig

if __name__ == "__main__":
    path_config = PathConfig.from_yaml("config_knn.yaml")
    main(path_config)
