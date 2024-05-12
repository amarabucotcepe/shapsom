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

import geopandas as gpd

import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout='wide')

df =  globals.current_database
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
    st.subheader('Seção 4 - Diferença entre grupos')
    st.text('etc')

def secao5():
    st.subheader('Seção 5 - Filtro de Triagem')
    st.text('Explicar como esse filtro está sendo aplicado e falar que ele também será aplicado na parte de anomalias')


st.title('Análise Por Grupos com SHAP/SOM')
secao1()
secao2()
secao3()
secao4()
secao5()





