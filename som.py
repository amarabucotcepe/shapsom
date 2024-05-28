import numpy as np
import warnings
warnings.filterwarnings("ignore")
import globals
from shaps import make_shap


def rodar_algoritmo():
        som_data = globals.som_data
        input_values = []
        for index, row in globals.crunched_df.iterrows():
            input_values.append(row[1:-1].values)

        make_shap(
            som_data['Municípios'], list(globals.crunched_df.columns[1:-1]), np.array(input_values, dtype=float),
            np.array(som_data['Nota']),
            globals.use_shap,
            desc=f"Gerando gráficos para {globals.crunched_df.columns[-1]}"
        )
        print(globals.shap_explanations)