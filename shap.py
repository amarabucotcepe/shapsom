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
import weasyprint
import imgkit

import matplotlib as mpl
from pypdf import PdfMerger