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
from sklearn.tree import DecisionTreeRegressor
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

st.info('Árvore de decisão', icon='⚔️')

# Define the features and the target
X = df[df.columns[3:-1]]
y = df[df.columns[-1]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor and fit it to the training data
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(X_train, y_train)

# Create a larger figure
fig, ax = plt.subplots(figsize=(20, 20))

# Plot the decision tree with larger fonts
tree.plot_tree(reg, ax=ax, feature_names=X.columns, filled=True, fontsize=10)

# Show the plot in Streamlit
st.pyplot(fig)