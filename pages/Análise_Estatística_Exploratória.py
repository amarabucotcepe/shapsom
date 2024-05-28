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
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import globals
import plotly.graph_objects as go

import geopandas as gpd

import os

# Set page configuration
st.set_page_config(layout='wide')

st.title("Relatório 📊")
st.subheader("Análise de dados")

title = st.text_input("Título do relatório")

# file = st.file_uploader("Faça upload do seu arquivo", type=['csv'])

# if file is not None:

df =  globals.current_database

# st.write(df)


st.info('Mapa da variável alvo', icon='🌎')
st.subheader('Mapa da variável alvo')

def generate_map():
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.read_file('PE_Municipios_2022.zip')
    gdf = gdf.merge(df[[df.columns[0],df.columns[-1]]], left_on='NM_MUN', right_on=df.columns[0])

    fig, ax = plt.subplots(1, 1)

    df[df.columns[-1]] = df[df.columns[-1]].round(2)

    m = gdf.explore(df.columns[-1], cmap='RdBu')

    components.html(m._repr_html_(), height=600)

    outfp = r"mapa.html"

    m.save(outfp)

with st.spinner('Gerando mapa...'):
    if os.path.exists('mapa.html'):
      m_repr_html_ = open('mapa.html').read()
      components.html(m_repr_html_, height=600)
    else:
        generate_map()

st.info(f'Município x {df.columns[-1]}', icon='🌎')

st.subheader('Análise Estatística')
# Calculate correlation
# dfmc = df.pivot_table(index=df.columns[0], values=df.columns[-1], aggfunc='mean')
dfmc = df.groupby(df.columns[0])[df.columns[-1]].apply(lambda x: x.mode().iloc[0]).reset_index()
# dfm = df.pivot_table(index=df.columns[0], values=df.columns[3:], aggfunc='mean')
dfm = df.groupby(df.columns[0])[df.columns[3:]].apply(lambda x: x.mode().iloc[0]).reset_index()

dfmc[dfmc.columns[-1]] = dfmc[dfmc.columns[-1]].round(2)
# dfm = df.pivot_table(index=df.columns[0], values=df.columns[3:-1], aggfunc=['mean','std'])
# dfm.columns = dfm.iloc[0]
# dfm = dfm[1:]
# st.write(dfm.head(5))


container = st.container(border=True)
container.write("O gráfico abaixo mostra a distribuição da variável resposta por município. Permite visualizar Municípios com valores extremos e dispersão em torno da média.")
st.markdown('Estatísticas')
st.dataframe(dfmc[dfmc.columns[-1]].describe().to_frame().T)

st.divider()

fig = px.scatter(
dfmc.reset_index(),
x="Município",
y=dfmc.columns[-1],
# size=dfmc.columns[-1],
hover_name="Município",
color=dfmc.columns[-1],
color_continuous_scale='icefire_r',
size_max=60,
)

fig.update_layout(
autosize=False,
width=800,
height=500,
shapes=[
    dict(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="Grey",
            width=1,
            )
        )
    ]
)

# Show the scatterplot in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.info(f'Variáveis por Município x {dfmc.columns[-1]}', icon='🌎')

container = st.container(border=True)
container.write("O gráfico abaixo mostra a relação da variável explicativa com a variável resposta. Permite visualizar como se correlacionam.")
with st.expander('ajuda',expanded=False):
    st.markdown('* $r = 1$:  correlação perfeita positiva, quanto maior o valor de uma variável, maior o valor da outra.')
    st.markdown('* $r = 0$:  não há correlação, não importa o valor de uma variável, o valor da outra não é afetado.')
    st.markdown('* $r = -1$:  correlação perfeita negativa, quanto maior o valor de uma variável, menor o valor da outra.')

corr = dfm[dfm.columns[3:-1]].corrwith(dfm[df.columns[-1]]).sort_values(ascending=False)

# Create a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm_r')

# Show the heatmap in Streamlit
st.pyplot(plt)

variavel = st.selectbox('Selecione a variável', df.columns[3:-1])
# Create a scatterplot of the penultimate column
fig = px.scatter(
dfm.reset_index(),
x=variavel,
y=dfmc.columns[-1],
# size=dfmc.columns[-1],
hover_name="Município",
color=variavel,
color_continuous_scale='icefire_r',
)

# Show the scatterplot in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.info('Correlações por Municipio', icon='⚔️')

#corr

# Create a heatmap
# fig = go.Figure(data=go.Heatmap(
#                 x=corr.index,
#                 y=['Correlation'],
#                 z=[corr.values],
#                 hoverongaps = False,
#                 colorscale='RdBu'))

# # Show the heatmap in Streamlit
# st.plotly_chart(fig)


with st.expander('Correlações por subunidade ⚔️',expanded=False):

    # Calculate correlation
    corr = df[df.columns[3:-1]].corrwith(df[df.columns[-1]]).sort_values(ascending=False)
    # corr

    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.index,
                    y=['0'],
                    hoverongaps = False,
                    colorscale='Viridis'))

    # Show the heatmap in Streamlit
    st.plotly_chart(fig)

    # Create a heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm_r')

    # Show the heatmap in Streamlit
    st.pyplot(plt)

st.subheader('Arvore de Decisão')
# Define the features and the target
X = df[df.columns[3:-1]]
y = df[df.columns[-1]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor and fit it to the training data
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(X_train, y_train)

# Create a pandas DataFrame with feature importances
feature_importances = pd.DataFrame(reg.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

st.info('Importância das variáveis', icon='📊')
# Display the feature importances in Streamlit
st.dataframe(feature_importances)

st.info('Árvore de decisão', icon='🌲')

# Create a larger figure
fig, ax = plt.subplots(figsize=(20, 20))

# Plot the decision tree with larger fonts
tree.plot_tree(reg, ax=ax, feature_names=X.columns, filled=True, fontsize=10)

# Show the plot in Streamlit
st.pyplot(fig)