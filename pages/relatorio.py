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

st.info(f'Município x {df.columns[-1]}', icon='🌎')


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
