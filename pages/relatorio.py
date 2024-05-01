from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import geopandas as gpd
from geopandas.tools import geocode
import geopandas.tools
from geopy.geocoders import Nominatim



import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout='wide')

st.title("Relatório 📊")
st.subheader("Análise de Agrupamento de dados")

title = st.text_input("Título do relatório")

# file = st.file_uploader("Faça upload do seu arquivo", type=['csv'])

# if file is not None:

file = 'Vacinação - UBS.csv'

df = pd.read_csv(file, sep=',')

st.write(df)

st.info('Dicionário de dados', icon='🎲')

# Get dataframe info
info_data = {
    'Column': df.columns,
    'Non-Null Count': df.count(),
    'Dtype': df.dtypes
}

info_df = pd.DataFrame(info_data).reset_index().drop('index', axis=1)

# Display the dataframe info as a table
st.table(info_df)

st.info('Mapa da variável alvo', icon='🌎')

# # Convert the DataFrame to a GeoDataFrame
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.geometry.str[1], df.geometry.str[0]))

# # Create an interactive map
# fig, ax = plt.subplots(figsize=(10, 10))
# gdf.plot(ax=ax, color='red')
# plt.show()

# # Show the map in Streamlit
# st.pyplot(fig)



st.info('Correlações por Municipio', icon='⚔️')

# Calculate correlation
dfm = df.pivot_table(index=df.columns[0], values=df.columns[3:-1], aggfunc='mean')
# dfm = df.pivot_table(index=df.columns[0], values=df.columns[3:-1], aggfunc=['mean','std'])
# dfm.columns = dfm.iloc[0]
# dfm = dfm[1:]
# dfm

corr = dfm[dfm.columns[3:-1]].corrwith(dfm[dfm.columns[-1]]).sort_values(ascending=False)
corr

# Create a heatmap
fig = go.Figure(data=go.Heatmap(
                x=corr.index,
                y=['Correlation'],
                hoverongaps = False,
                colorscale='Viridis'))

# Show the heatmap in Streamlit
st.plotly_chart(fig)

# Create a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')

# Show the heatmap in Streamlit
st.pyplot(plt)


with st.expander('Correlações por UBS ⚔️',expanded=False):


    # Calculate correlation
    corr = df[df.columns[3:-1]].corrwith(df[df.columns[-1]]).sort_values(ascending=False)
    corr

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
    sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')

    # Show the heatmap in Streamlit
    st.pyplot(plt)