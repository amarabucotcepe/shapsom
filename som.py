import numpy as np
import warnings
warnings.filterwarnings("ignore")
import globals
from shaps import make_shap
import copy


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
    # print(globals.shap_explanations)
    # print(globals.shap_columns)
    
    
    # maiores_valores_colunas = np.zeros(len(globals.shap_columns))
    # menores_valores_colunas = np.zeros(len(globals.shap_columns))
        
    # for coluna in range(len(globals.shap_columns)):
    #     for i in range(len(globals.shap_explanations)):
            
            # if (globals.shap_explanations[i].values[coluna] > 0):
                
            #     if (globals.shap_explanations[i].values[coluna] > maiores_valores_colunas[coluna]):
            #         maiores_valores_colunas[coluna] = globals.shap_explanations[i].values[coluna]
            # else:
            #     if (globals.shap_explanations[i].values[coluna] < menores_valores_colunas[coluna]):
            #         menores_valores_colunas[coluna] = globals.shap_explanations[i].values[coluna]
        
    
    # print(maiores_valores_colunas)
    # print(menores_valores_colunas)
    
    min = float('inf')
    max = float('-inf')
    
    for i in range(len(globals.shap_explanations)):
        valor = globals.shap_explanations[i].values
        if(np.max(valor) > max):
            max = np.max(valor)
        if(np.min(valor) < min):
            min = np.min(valor)
            


    
    globals.som_data['SHAP Original'] = np.zeros(len(globals.som_data))
    globals.som_data['SHAP Normalizado'] = np.zeros(len(globals.som_data))
    
    globals.som_data['SHAP Original'] = globals.som_data['SHAP Original'].apply(lambda x: [x])
    globals.som_data['SHAP Normalizado'] = globals.som_data['SHAP Normalizado'].apply(lambda x: [x])
    
    for i in range(len(globals.shap_explanations)):
        array_normalizado = []
        for coluna in range(len(globals.shap_columns)):
            valor = globals.shap_explanations[i].values[coluna]
            if(valor > 0):
                array_normalizado.append(normalizar_entre_dois_valores(valor, 0, max, 0, 1))
            elif(valor < 0):
                array_normalizado.append(normalizar_entre_dois_valores(valor, min, 0 , -1, 0))
            else:
                array_normalizado.append(0)
        
        globals.som_data['SHAP Original'][i] = globals.shap_explanations[i].values
        globals.som_data['SHAP Normalizado'][i] = array_normalizado
    
    resultado = globals.som_data.groupby('Grupo')['SHAP Normalizado'].apply(lambda x: [sum(i)/len(i) for i in zip(*x)])
    
    globals.som_data['SHAP Media Cluster'] = globals.som_data['Grupo'].map(resultado)
        
               
def normalizar_entre_dois_valores(valor, min, max, a, b):
    return a + ((valor - min) * (b - a) / (max - min))