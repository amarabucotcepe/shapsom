from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

import branca.colormap as cm
import folium
import json

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

# Calculate correlation
dfmc = df.pivot_table(index=df.columns[0], values=df.columns[-1], aggfunc='mean')
dfm = df.pivot_table(index=df.columns[0], values=df.columns[3:-1], aggfunc='mean')
# dfm = df.pivot_table(index=df.columns[0], values=df.columns[3:-1], aggfunc=['mean','std'])
# dfm.columns = dfm.iloc[0]
# dfm = dfm[1:]
# dfm

st.info('Distribuição da variável alvo', icon='🌎')

# Create a scatterplot of the penultimate column
fig = go.Figure(data=go.Scatter(x=dfmc.index, y=dfmc[dfmc.columns[-1]], mode='markers'))

fig.update_layout(
    title='Scatterplot of ' + df.columns[-1],
    xaxis_title='Index',
    yaxis_title=dfmc.columns[-1]
)

# Show the scatterplot in Streamlit
st.plotly_chart(fig)

st.info('Correlações por Municipio', icon='⚔️')
with st.expander('ajuda',expanded=False):
    st.markdown('* 1= correlação perfeita positiva, quanto maior o valor de uma variável, maior o valor da outra.')
    st.markdown('* 0= não há correlação, não importa o valor de uma variável, o valor da outra não é afetado.')
    st.markdown('* -1= correlação perfeita negativa, quanto maior o valor de uma variável, menor o valor da outra.')


corr = dfm[dfm.columns[3:-1]].corrwith(dfm[dfm.columns[-1]]).sort_values(ascending=False)

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

st.divider()

# Create a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')

# Show the heatmap in Streamlit
st.pyplot(plt)


with st.expander('Correlações por UBS ⚔️',expanded=False):

    # Calculate correlation
    corr = df[df.columns[3:-1]].corrwith(df[df.columns[-1]]).sort_values(ascending=False)
    # corr

    # # Create a heatmap
    # fig = go.Figure(data=go.Heatmap(
    #                 z=corr.values,
    #                 x=corr.index,
    #                 y=['0'],
    #                 hoverongaps = False,
    #                 colorscale='Viridis'))

    # # Show the heatmap in Streamlit
    # st.plotly_chart(fig)

    # Create a heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')

    # Show the heatmap in Streamlit
    st.pyplot(plt)
    
    
    # Create a boxplot of the last column grouped by the first column
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[df.columns[0]], y=df[df.columns[-1]])

    plt.title('Boxplot of ' + df.columns[-1] + ' by ' + df.columns[1])
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[-1])

    # Show the boxplot in Streamlit
    st.pyplot(plt)

st.info('Árvore de decisão', icon='⚔️')
