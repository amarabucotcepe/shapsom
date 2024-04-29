import streamlit as st

from scipy.cluster.hierarchy import linkage, fcluster
from colorsys import hsv_to_rgb
from minisom import MiniSom
import numpy as np
import pandas as pd
import os
import re
import io
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tqdm import tqdm
import shap
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import json
import folium
import shutil
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from difflib import SequenceMatcher
import statistics
import math
from sklearn.ensemble import IsolationForest

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from reportlab.platypus.flowables import Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from unidecode import unidecode
import imgkit

import matplotlib as mpl
from pypdf import PdfMerger
from globals import selected_df, cluster_distance, epochs, size, sigma,lr, use_shap, current_database, current_database_name, current_output_columns, current_hidden_columns, current_input_columns, current_label_columns

from som import rodar_algoritmo
# from shap import 
from report import (
documento_1,
documento_2_2_e_2_4,
documento_2_3,
documento_2_5,
documento_2_6,
documento_7_1,
salvar_pdfs,
add_cabecalho,
gerar_anexos
)

"""### Anexos"""

gerar_anexos()

st.title("ShapSom")
st.subheader("Análise de agrupamento de dados")

title = st.text_input("Título do relatório")

file = st.file_uploader("Faça upload do seu arquivo", type=['csv'])

if file is not None:

    if file.name.endswith(".csv"):
        df = pd.read_csv(file, sep=',')
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    # st.dataframe(df)

    current_database = df.dropna()
    current_database_name = file.name.split(".")[0]

    string_list = df.columns.tolist()

    st.divider()

    current_label_columns = []
    current_input_columns = []
    current_output_columns = []
    current_hidden_columns = []

    st.subheader("Selecione as colunas")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            current_label_columns = st.multiselect("Nome", string_list)
        with col2:
            options = [col for col in string_list if col not in current_label_columns]
            current_input_columns = st.multiselect("Entradas", [col for col in string_list if col not in current_label_columns])
        with col3:
            current_output_columns = st.selectbox("Saída", [col for col in string_list if col not in current_label_columns and col not in current_input_columns])
        with col4:
            current_hidden_columns = st.multiselect("Ocultar", [col for col in string_list if col not in current_label_columns and col not in current_input_columns and col not in current_output_columns])

    selected_df = df.drop(columns=current_hidden_columns)

    st.divider()

    # st.subheader("Selecione os dados de interesse")
    # if df[saida].min() is not None:
    #     triagem = st.slider("Triagem", min_value=df[saida].min(), max_value=df[saida].max(), value=(df[saida].min(), df[saida].max()), step=0.1)
    #     selected_df = selected_df.loc[(selected_df[saida] > triagem[0]) & (df[saida] < triagem[1])]
    

    with st.expander("Parâmetros SOM", expanded=False):
        cluster_distance = st.slider("Cluster Distance", min_value=0.5, max_value=4.0, value=1.0, step=0.1)
        epochs = st.slider("Epochs", min_value=2, max_value=5, value=3, step=1)  # Changed step to int
        size = st.slider("Size", min_value=5, max_value=60, value=30, step=1)  # Changed step to int
        sigma = st.slider("Sigma", min_value=1, max_value=10, value=9, step=1)  # Changed step to int
        lr = st.slider("Learning Rate", min_value=-3.0, max_value=-1.0, value=-2.0, step=0.1)
    
    
    use_shap = st.checkbox("Criar SHAP",help='Selecione para obter análise completa dos dados')

    submit_button = st.button('Executar')
    
    
    if submit_button:
        st.dataframe(selected_df)

        rodar_algoritmo()
        # documento_1()
        # documento_2_2_e_2_4()
        # documento_2_3()
        # documento_2_5()
        # documento_2_6()
        # documento_7_1()
        # salvar_pdfs()
        # add_cabecalho(title)
        # gerar_anexos()