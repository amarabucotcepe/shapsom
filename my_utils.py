import pandas as pd
import numpy as np
import random
import string
from minisom import MiniSom
from scipy.cluster.hierarchy import linkage, fcluster
import colorsys
import warnings
warnings.filterwarnings("ignore")

HEX_SHAPE = "M0,-1 L0.866,-0.5 L0.866,0.5 L0,1 L-0.866,0.5 L-0.866,-0.5 Z"

def cluster_coordinates(coordenadas: 'list[tuple]', distancia_maxima: float) -> 'list[list[tuple]]':
    '''Recebe uma lista de coordenadas e devolve elas agrupadas de acordo com a distância definida'''
    mat = linkage(coordenadas, method='single', metric='chebyshev')
    cluster_ids = fcluster(mat, distancia_maxima, criterion='distance')
    elements_with_id = lambda id : np.array(np.where(cluster_ids == id), dtype=int).flatten().tolist()
    clusters = [[coordenadas[i] for i in elements_with_id(id)] for id in set(cluster_ids)]
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters

def hsv_to_hex(hsv) -> str:
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    r,g,b = int(r * 255), int(g * 255), int(b * 255)
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    return hex_color

def random_string(N: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=N))

def normalize(data: np.ndarray) -> np.ndarray:
    data = (data if isinstance(data, np.ndarray) else np.array(data)).astype(np.float64)
    data -= data.min()
    data /= data.max()
    return data

def get_som_data(som: MiniSom, labels, x, y, cluster_distance) -> pd.DataFrame:
    units = som.labels_map(x, labels)
    is_hex = som.topology == "hexagonal"
    units = {((_x + 0.5) if (is_hex and _y % 2 == 1) else (_x), _y) : units[_x,_y] for _x,_y in units.keys()}
    clusters = cluster_coordinates(list(units.keys()), cluster_distance)
    
    cluster_amount = len(clusters)
    hsv_boundaries = ((np.arange(cluster_amount+1) / (cluster_amount+1)) * 360).flatten().tolist()
    hsv_region_width = (360 / (cluster_amount+1)) * 0.25
    hsv_centers = np.array([(hsv_boundaries[i] + hsv_boundaries[i+1]) / 2 for i in range(cluster_amount)]).flatten().tolist()
    hsv_regions = np.array([(c-(hsv_region_width/2), c+(hsv_region_width/2)) for c in hsv_centers]).tolist()
    som_data_list = []

    for i, coords in enumerate(clusters):
        coord_dict = {}
        cell_scores = []
        for c in filter(lambda _c : units[_c], coords):
            cell_labels = list(units[c])
            score_data = np.array([y.loc[y['label'] == u, y.columns[-1]].values[0] for u in cell_labels]).flatten()
            variables = [x[list(labels).index(v)] for v in units[c]]
            cell_score = np.average(score_data)
            cell_scores.append(cell_score)
            coord_dict[c] = {"labels": cell_labels, "scores": score_data, "variables": variables, "cell_score": cell_score}
        
        min_hsv, max_hsv = hsv_regions[i]
        min_score, max_score = min(cell_scores), max(cell_scores)
        central_color = hsv_to_hex([(min_hsv+max_hsv / 2), 1, 1])
        
        for c in coord_dict.keys():
            for municipio in coord_dict[c]["labels"]:
                #_labels = ", ".join(coord_dict[c]["labels"])
                _labels = municipio
                _score = coord_dict[c]["cell_score"]
                _x = c[0]
                _y = c[1]
                _color_percent = 0.5 if (min_score == max_score) else (_score - min_score) / (max_score - min_score)
                _hue_hsv = (max_hsv * _color_percent) + (min_hsv * (1 - _color_percent))
                _color_hex = hsv_to_hex([_hue_hsv, 1, 1])
                som_data_list.append([_labels, round(_score, 2), _x, _y, _color_hex, central_color, i+1])
    
    som_data = pd.DataFrame(som_data_list, columns=["labels", "score", "x", "y", "color", "central_color","cluster"])
    return som_data

def create_map(df: pd.DataFrame, label_column: str, variable_columns: 'list[str]', output_column: str, size: int = 50, lr: float = 1e-1, epochs: int = 1000, sigma = 2, topology = "hexagonal", cluster_distance: float = 2, interval_epochs: int=100, output_influences=True):
    labels = df[label_column]

    x = np.array(df[variable_columns].select_dtypes(include='number').values)
    if output_influences:
        val_multipliers = df[output_column].values
        new_x = []
        for r,m in zip(x, val_multipliers):
            new_r = r if np.sum(r) != 0 else np.ones_like(r)
            new_r /= np.sum(new_r)
            new_x.append(new_r * m)
        x = np.array(new_x)
        
    y = pd.concat([pd.DataFrame({"label": labels}), df[output_column]], axis=1)
    som = MiniSom(
        x=size,
        y=size,
        input_len=len(x[0]),
        sigma=sigma,
        topology=topology,
        learning_rate=lr,
        neighborhood_function="gaussian",
        activation_distance="euclidean"
    )
    
    som.pca_weights_init(x)
    for _ in range(epochs // interval_epochs):
        som.train(x, interval_epochs, verbose=False)
        som_data = get_som_data(som, labels, x, y, cluster_distance)
        yield som_data

def verificarColunaDesc(database):
    nStrings = 0
    for i in range(database.shape[1]):
        try:
            float(np.array(database.iloc[0])[i])
        except ValueError:
            nStrings+=1
    return nStrings==database.shape[1]
  
def convert_numeric(x):
    try:
        return np.int64(float(x)) if float(x).is_integer() else np.float64(x)
    except (ValueError,AttributeError):
        return x