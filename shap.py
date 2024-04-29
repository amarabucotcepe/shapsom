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

def make_shap(labels, variable_columns, x, y, use_shap, desc="Gerando gráficos SHAP"):
    model = XGBRegressor(objective='reg:squarederror', random_state= 34)
    x *= 10
    y *= 10
    model.fit(x, y)
    global explanations
    global shap_labels
    global shap_columns
    global shape_results
    shap_columns = variable_columns
    shap_labels = labels
    explainer = shap.Explainer(model, x)
    explanations = [shap.Explanation(values=v, base_values=explainer.expected_value, feature_names=variable_columns) for v in explainer(x)]

    # Salva os mapas um por um e exibe uma barrinha de progresso para o usuário
    if use_shap:
        for i, exp in tqdm(enumerate(explanations), desc=desc, total=len(explanations)):
            fig = plt.figure()
            shap.waterfall_plot(exp, show=False)
            plt.title(labels[i])
            fig.set_size_inches(16, 8)
            fig.subplots_adjust(left=0.4)
            plt.close(fig)
            shape_results[labels[i].split(" - ")[0]] = {}
            shape_results[labels[i].split(" - ")[0]]['data'] = exp.data.copy()
            shape_results[labels[i].split(" - ")[0]]['values'] = exp.values.copy()
