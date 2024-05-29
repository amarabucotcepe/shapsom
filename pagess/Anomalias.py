from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px

import branca.colormap as cm
from branca.colormap import linear

import folium
import json
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import globals
import math
import geopandas as gpd

import plotly.graph_objects as go

# Set page configuration
#st.set_page_config(layout='wide')

def pagina_anomalias():
    st.title('Seção de Anomalias')
    st.write("""A análise de anomalias foi conduzida utilizando um Mapa Auto-Organizável (SOM) para identificar pontos de dados que se desviam significativamente do padrão observado.
    Com as coordenadas dos pontos no SOM, o centroide do mapa foi calculado. Este centroide é determinado utilizando a mediana das coordenadas x e y de todos os pontos, o que fornece uma medida menos sensível a outliers em comparação com a média. Então, são calculadas as distâncias dos pontos para o centroide do mapa.
    Pontos que apresentaram distâncias significativamente maiores em relação ao centroide foram identificados como anômalos. Estes pontos fora do cluster principal sugerem comportamentos ou características discrepantes dos dados normais, destacando-se por estarem afastados do padrão usual.""")

    has_databases = True
    try:
        has_databases = has_databases and globals.som_chart is not None
        has_databases = has_databases and globals.som is not None
    except:
        has_databases = False
    
    st.divider()
    if not has_databases:
        st.write("Nenhum dataset foi carregado. Por favor, carregue um dataset e tente novamente.")
    else:
        globals.som = st.altair_chart(globals.som_chart, use_container_width=True)
        df = globals.som_data
        med_x = df['x'].median()
        med_y = df['y'].median()
        st.write(f"O centroide do está localizado em: x = {med_x} e y = {med_y}")
        globals.porcentagem = st.slider("Porcentagem", min_value=1, max_value=100, step=1, value=10, help="Porcentagem de anomalias esperadas. Por exemplo, se o valor for 10, o algoritmo irá mostrar 10% dos dados como anomalias.")
        get_anomalies = st.button("Obter Anomalias")
        if get_anomalies:
            porcentagem=10
            distancias = {}
            with st.spinner('Calculando distâncias...'):
                for i in range(df.shape[0]):
                    x = df['x'][i]
                    y = df['y'][i]
                    dist = math.sqrt(math.pow(x-med_x,2)+math.pow(y-med_y,2))
                    distancias[i] = dist
                dist_df = pd.DataFrame({'Distância do centroide':pd.Series(distancias)})
                df_aux = dist_df.join(df)
                df_aux = df_aux.sort_values(by=['Distância do centroide'], ascending=False).head(df.shape[0]//porcentagem)
                df_aux = df_aux.drop(['Cor'],axis=1).reset_index(drop=True)
                st.dataframe(df_aux, use_container_width=True)