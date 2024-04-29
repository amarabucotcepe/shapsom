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

from globals import current_label_columns, cluster_distance, current_database, current_database_name, current_hidden_columns, current_input_columns, current_output_columns,epochs, size, sigma, lr, use_shap 
from shap import make_shap

def normalize(data: np.ndarray | list) -> np.ndarray:
    data = (data if isinstance(data, np.ndarray) else np.array(data)).astype(np.float64)
    data -= data.min()
    data /= data.max()
    return data

def cluster_coordinates(coordenadas: list[tuple], distancia_maxima: float) -> list[list[tuple]]:
    mat = linkage(coordenadas, method='single', metric='chebyshev')
    cluster_ids = fcluster(mat, distancia_maxima, criterion='distance')
    elements_with_id = lambda id : np.array(np.where(cluster_ids == id), dtype=int).flatten().tolist()
    clusters = [[coordenadas[i] for i in elements_with_id(id)] for id in set(cluster_ids)]
    return clusters

def create_map(
    title: str,
    df: pd.DataFrame,
    label_columns: list[str],
    variable_columns: list[str],
    output_columns: list[str],
    size: int = 50,
    lr: float = 1e-1,
    epochs: int = 1000,
    sigma = 2,
    cluster_distance: float = 2,
    use_shap = False
    ):
    # Mapa SOM
    print("chega aq", label_columns)
    labels = df[label_columns].apply(lambda row: ' - '.join(map(str, row)), axis=1)
    x = df[variable_columns].select_dtypes(include='number').values
    y = pd.concat(
        [pd.DataFrame({"label": labels, "Média dos dados": x.mean(axis=1)}), df[output_columns].select_dtypes(include='number')],
        axis=1
    )
    output_columns += ["Média dos dados"]

    som = MiniSom(
        x=size,
        y=size,
        input_len=len(x[0]),
        sigma=sigma,
        topology="hexagonal",
        learning_rate=lr,
        neighborhood_function="gaussian",
        activation_distance="euclidean"
    )
    som.pca_weights_init(x)
    print("Treinando mapa SOM...")
    som.train(x, epochs, verbose=True)

    distance_map = normalize(som.distance_map().T)
    units = som.labels_map(x, labels)
    clusters = cluster_coordinates(list(units.keys()), cluster_distance)
    global cluster_dict

    # Coleta as informações de cada cluster
    for i, coords in enumerate(clusters):
        coord_dict = {}
        for c in filter(lambda _c : units[_c], coords):
            cell_labels = list(units[c])
            score_dict = {}
            cell_score_dict = {}
            for score_type in y.columns[1:]:
                data = [y.loc[y['label'] == u, score_type].values[0] for u in cell_labels]
                score_dict[score_type] = data
            for k in score_dict.keys():
                cell_score_dict[k] = np.average(score_dict[k])

            variables = [x[list(labels).index(v)] for v in units[c]]

            coord_dict[c] = {
                "labels": cell_labels,
                "scores": score_dict,
                "variables": variables,
                "height": distance_map[c],
                "cell_scores": cell_score_dict
            }

        score_dict = {}
        for c in coord_dict.keys():
            for score_type in coord_dict[c]["scores"].keys():
                if not score_type in score_dict.keys():
                    score_dict[score_type] = []
                score_dict[score_type] += coord_dict[c]["scores"][score_type]

        c_score = {}
        for k in score_dict.keys():
            c_score[k] = np.average(score_dict[k])

        for c in coord_dict.keys():
            coord_dict[c]["cluster_scores"] = c_score

        cluster_dict[f"cluster {i+1}"] = coord_dict

    # SHAP
    #if use_shap:
    for y_label in output_columns:
        labels = []
        variables = []
        scores = []

        for cluster_name in cluster_dict.keys():
            cluster = cluster_dict[cluster_name]
            coord_list = [cluster[coord] for coord in cluster.keys()]

            for coord in coord_list:
                labels += coord["labels"]
                variables += coord["variables"]
                scores += coord["scores"][y_label]

        labels = [f"{label} - {y_label}" for label in labels]
        make_shap(
            labels, variable_columns, np.array(variables),
            np.array(scores),
            use_shap,
            desc=f"Gerando gráficos para {y_label}"
        )


def rodar_algoritmo():
        create_map(
        current_database_name,
        deepcopy(current_database),
        cluster_distance=cluster_distance,
        epochs=epochs,
        size=size,
        sigma=sigma,
        lr=lr,
        label_columns=current_label_columns,
        variable_columns=list(filter(lambda c : c in current_database.columns, current_input_columns)),
        output_columns=deepcopy(current_output_columns),
        use_shap=use_shap
    )