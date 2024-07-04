from songdo_traffic_core.dataset.interpolator import IterativeRandomForestInterpolator
from songdo_traffic_core.dataset.metr_imc.generator import MetrImcSubsetGenerator
import pandas as pd

df_imc: pd.DataFrame = pd.read_hdf("./datasets/metr-imc/metr-imc.h5")

no_missing_columns = df_imc.columns[df_imc.isnull().sum() == 0].to_list()
less_500_missing_columns = df_imc.columns[df_imc.isnull().sum() < 500].to_list()
less_750_missing_columns = df_imc.columns[df_imc.isnull().sum() < 750].to_list()

generator = MetrImcSubsetGenerator(
    nodelink_dir="./datasets/metr-imc/nodelink",
    imcrts_dir="./datasets/metr-imc/imcrts",
    metr_imc_dir="./datasets/metr-imc/",
)

interpolator = IterativeRandomForestInterpolator(verbose=50)

generator.generate_subset(
    less_500_missing_columns, "./datasets/metr-imc-296", interpolator,
)
