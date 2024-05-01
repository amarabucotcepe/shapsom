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

import geopandas as gpd

import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout='wide')

st.title("Relatório 📊")
st.subheader("Análise de dados")

title = st.text_input("Título do relatório")

# file = st.file_uploader("Faça upload do seu arquivo", type=['csv'])

# if file is not None:

file = 'Vacinação - UBS.csv'

df = pd.read_csv(file, sep=',')

# st.write(df)

with st.expander('Dicionário de dados 🎲',expanded=False):
    # Get dataframe info
    info_data = {
        'Column': df.columns,
        'Non-Null Count': df.count(),
        'Dtype': df.dtypes
    }

    info_df = pd.DataFrame(info_data).reset_index().drop('index', axis=1)

    # Display the dataframe info as a table
    st.table(info_df)

st.info('Mapa da variável alvo', icon='🌎')

# # Convert the DataFrame to a GeoDataFrame
gdf = gpd.read_file('PE_Municipios_2022.zip')
gdf = gdf.merge(df[[df.columns[0],df.columns[-1]]], left_on='NM_MUN', right_on=df.columns[0])

fig, ax = plt.subplots(1, 1)

df[df.columns[-1]] = df[df.columns[-1]].round(2)

m = gdf.explore(df.columns[-1], cmap='RdBu')

components.html(m._repr_html_(), height=600)

outfp = r"mapa.html"

m.save(outfp)

