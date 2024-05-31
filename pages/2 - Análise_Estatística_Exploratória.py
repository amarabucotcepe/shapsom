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

st.title("Análise Estatística Exploratória 📊")
st.subheader("Análise de dados")


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
st.info('Figura 1 - Mapa do Estado de Pernambuco')

st.markdown(''' A figura 1 exibe uma análise geoespacial das notas de proficiência dos estudantes,  no âmbito da educação básica. As diferentes tonalidades de cores no mapa indicam variações nos níveis de proficiência entre os diversos municípios. A imagem, portanto, oferece uma visão detalhada e regionalizada do desempenho educacional no estado de Pernambuco, evidenciando as disparidades entre diferentes municípios.
Este tipo de visualização é essencial para identificar regiões que necessitam de intervenções educacionais mais intensivas e ajuda a entender as disparidades educacionais entre diferentes regiões, contribuindo para políticas públicas mais informadas e direcionadas. 
''')

with st.container(border=True) as container1:
    st.markdown(''' **Interpretação do Mapa** \n
        Tonalidades de Cores: \n
        -Cores Vermelhas: Indicam áreas com menores notas de proficiência, sugerindo um desempenho educacional mais baixo.\n
        -Cores Azuis: Representam áreas com maiores notas de proficiência, indicando um desempenho educacional superior.\n
        -Cores Mais Escuras: Correspondem aos extremos dos níveis de proficiência, onde as cores vermelhas mais escuras representam os piores desempenhos e as cores azuis mais escuras representam os melhores desempenhos.\n''')
    
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




st.markdown('Estatísticas')
st.dataframe(dfmc[dfmc.columns[-1]].describe().to_frame().T)

st.divider()

st.markdown("O gráfico abaixo mostra a distribuição da variável resposta por município. Permite visualizar Municípios com valores extremos e dispersão em torno da média.")
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
st.info('Gráfico 1: gráfico de dispersão da distribuição das notas de proficiência dos estudantes por municípios')
st.markdown(''' O gráfico 1 faz parte de uma análise estatística mais ampla apresentada no relatório, que visa explorar a variabilidade e o desempenho educacional em todo o estado. O gráfico 1 permite identificar visualmente quais municípios apresentam desempenhos extremos, tanto positivos quanto negativos, e como as notas estão dispersas em relação à média. Esta visualização facilita a identificação de áreas que necessitam de maior atenção e recursos.
''')
with st.container(border=True) as container2:
    st.markdown(''' **Análise do Gráfico de Dispersão** \n
    **Eixos do Gráfico** \n
    -Eixo X: Representa os municípios de Pernambuco. Cada ponto ao longo deste eixo corresponde a um município específico.\n
    -Eixo Y: Representa as notas de proficiência dos estudantes, variando de aproximadamente 0,43 a 0,70.\n
    **Distribuição das Notas:** \n
    -As diferentes cores dos pontos no gráfico indicam os níveis de proficiência, conforme a legenda de cores à direita.\n
    -Cores Azuis: Indicam os municípios com as melhores notas de proficiência, situando-se entre 0,60 e 0,70.\n
    -Cores Vermelhas: Indicam os municípios com as piores notas de proficiência, situando-se entre 0,43 e 0,50.
    ''')

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

st.info('Figura 2: Árvore de Decisão')
st.markdown(''' **Análise da Árvore de Decisão** \n
A figura 2 apresenta uma análise com base em uma árvore de decisão, uma ferramenta de aprendizado de máquina usada para tomar decisões baseadas em dados. O objetivo da árvore de decisão é modelar e prever um valor de saída com base em diversas variáveis de entrada. \n
Na parte superior da figura 2, temos uma tabela que lista a importância das variáveis utilizadas no modelo. A importância de uma variável é uma medida de quanto essa variável contribui para a predição do modelo. Essa tabela indica que a variável SAEPE - Participação é a mais importante para o modelo, seguida pela Adesão ao Programa Criança Alfabetizada e Adesão ao Compromisso Nacional Criança Alfabetizada. Outras variáveis: 0 (não tiveram contribuição significativa no modelo). 
''')
st.markdown(''' **Interpretação da Árvore de Decisão** \n
A árvore de decisão está representada graficamente logo abaixo da tabela. Para interpretar a árvore, siga estas etapas:\n
-Raiz da Árvore: A árvore começa com a variável mais importante, que é a SAEPE - Participação. O nó raiz divide os dados em duas partes com base em um valor de limiar para esta variável (<= 0.945).\n
-Nódulos e Ramificações: Cada nó da árvore representa uma decisão baseada em um valor limiar de uma variável. Por exemplo, no segundo nível da árvore, para os dados que têm SAEPE - Participação menor ou igual a 0.945, a próxima divisão é baseada na Adesão ao Compromisso Nacional Criança Alfabetizada com um valor de corte de 0.5.\n
-Folhas: As folhas da árvore (os nós finais) representam as predições finais do modelo. Cada folha mostra a média dos valores de saída dos exemplos que chegaram a esse nó, o erro quadrado médio (squared error), o número de amostras (samples), e o valor previsto (value).\n

A árvore de decisão mostra que a SAEPE - Participação é a variável mais influente na predição do modelo, seguida pela Adesão ao Programa Criança Alfabetizada e Adesão ao Compromisso Nacional Criança Alfabetizada. As outras variáveis não tiveram impacto significativo nas previsões. A análise das subdivisões e folhas finais da árvore ajuda a entender como diferentes níveis de participação e adesão aos programas influenciam o valor previsto.
''')