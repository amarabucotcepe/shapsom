from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import numpy as np
import branca.colormap as cm
from branca.colormap import linear
import re
import folium
import json
from streamlit_folium import st_folium
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import globals

import geopandas as gpd
import os

import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout='wide')

df =  globals.current_database

for i in range(globals.som_data['Grupo'].max() + 1):
    filepath = f'mapa{i}.html'
    if os.path.exists(filepath):
        os.remove(filepath)
        
def secao1():
    st.subheader('Seção 1 - Dicionário de Dados')
    st.text('(explicar o que é o dicionário de Dados)')

def secao2():
    st.subheader('Seção 2 - Visão Geral de Dados e Heatmap')
    st.text('(Explicar o que é o Heatmap e como está sendo feita a média e o desvio padrão deles)')
    
def secao3():
    st.subheader('Seção 3 - Análise entre grupos')
    st.text('Explicar como os grupos são formados (explicar o SOM e o conceito de clusters etc) e explicar os valores da tabela')

def secao4():
    #Criando as variáveis
    original_df = globals.crunched_df
    df = globals.som_data
    max_grupo = df['Grupo'].max()
    df_expandido = df.assign(municipios=df['Municípios'].str.split(',')).explode('municipios').reset_index(drop=True)
    df_expandido = df_expandido.drop(columns=['Municípios', 'x', 'y'])
    grupos = df_expandido.groupby('Grupo')

            
    st.subheader('Seção 4 - Diferença entre grupos')
    st.markdown('''A análise comparativa entre os agrupamentos é conduzida combinando todas as informações 
                da "Análise de Agrupamento" (Seção 3), organizando-as em uma disposição paralela. Isso tem o 
                objetivo de destacar de forma mais clara as disparidades nas estruturas dos agrupamentos.''')    

    for i in range(max_grupo+1):
       if i in grupos.groups:
        #Tabelas
        st.info(f'Grupo {i}')
        grupo_df = grupos.get_group(i)
        cor_grupo = grupo_df['Cor'].iloc[0]
        lista_cores = grupo_df['Cor'].tolist()

        def apply_color(val):
            return f"background-color: {cor_grupo}; "

        st.dataframe(grupo_df.style.applymap(apply_color).format({'Nota':'{:.2f}'}), column_order=['municipios', 'Nota', 'Grupo', 'Cor'] ,column_config={
            'municipios': 'Municípios',
            'Nota': 'Nota do Município',
            'Grupo': 'Grupo',
            'Cor': None
            }             
        )

        #Mapas
        
        def generate_map():
            # Convert the DataFrame to a GeoDataFrame
            gdf = gpd.read_file('PE_Municipios_2022.zip')
            gdf = gdf.merge(grupo_df[[grupo_df.columns[2],grupo_df.columns[-1]]], left_on='NM_MUN', right_on=grupo_df.columns[-1])

            fig, ax = plt.subplots(1, 1)

            custom_cmap = mcolors.ListedColormap([cor_grupo])
            
            values_range = np.linspace(0, 1, 10)

            # Plot the map and apply the custom colormap
            m = gdf.explore(column=grupo_df.columns[2], cmap=custom_cmap, vmin=0, vmax=1)

            components.html(m._repr_html_(), height=600)

            outfp = f"mapa{i}.html"

            m.save(outfp)

        with st.spinner('Gerando mapa...'):
            if os.path.exists(f'mapa{i}.html'):
                m_repr_html_ = open(f'mapa{i}.html').read()
                components.html(m_repr_html_, height=600)
            else:
                generate_map()

def secao5():
    st.subheader('Seção 5 - Filtro de Triagem')
    st.text('Explicar como esse filtro está sendo aplicado e falar que ele também será aplicado na parte de anomalias')

st.title('Análise Por Grupos com SHAP/SOM')
secao1()
secao2()
secao3()
secao4()
secao5()





