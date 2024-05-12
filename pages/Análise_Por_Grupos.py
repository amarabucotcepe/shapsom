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

st.title("Relatório 📊")
st.subheader("Análise de dados")

title = st.text_input("Título do relatório")

# file = st.file_uploader("Faça upload do seu arquivo", type=['csv'])

# if file is not None:

df =  globals.current_database




