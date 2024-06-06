from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px

import branca.colormap as cm
from branca.colormap import linear
from PIL import Image
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
def quebra_pagina():
    st.markdown("""
        <style type="text/css" media="print">
        hr{
            page-break-after: always;
            page-break-inside: avoid;
        }
        <style>
    """, unsafe_allow_html= True)
    
def pagina_analise_estatistica_exploratoria():
    st.title("Relatório 📊")
    st.subheader("Análise Estatística Exploratória")

    has_databases = True
    try:
        has_databases = has_databases and globals.current_database is not None
    except:
        has_databases = False

    if has_databases:

        # st.write(df)

        df =  globals.current_database
        st.subheader('Mapa de Análise da Variável Alvo')

        st.markdown('''O mapa de análise da variável alvo apresenta uma análise geoespacial dos municípios do estado de Pernambuco. As diferentes tonalidades de cores no 
                    mapa representam as variações nos níveis da variável de escolha. As áreas em tons mais escuros indicam um desempenho superior, 
                    enquanto as áreas em tons mais claros refletem um desempenho inferior. Esta visualização detalhada é crucial para identificar regiões que necessitam de 
                    intervenções mais intensivas, ajudando a direcionar políticas públicas e recursos de forma mais eficiente.''')

        botao_mapa = st.button('Gerar mapa')
        
        if botao_mapa:
            def generate_map():
                # Convert the DataFrame to a GeoDataFrame
                gdf = gpd.read_file('PE_Municipios_2022.zip')
                gdf = gdf.merge(df[[df.columns[0],df.columns[-1]]], left_on='NM_MUN', right_on=df.columns[0])

                fig, ax = plt.subplots(1, 1)

                df[df.columns[-1]] = df[df.columns[-1]].round(2)

                m = gdf.explore(df.columns[-1], cmap='RdBu', fitbounds="locations")

                components.html(m._repr_html_(), height=400)

                outfp = r"mapa.html"

                m.save(outfp)

            with st.spinner('Gerando mapa...'):
                if os.path.exists('mapa.html'):
                    m_repr_html_ = open('mapa.html').read()
                    components.html(m_repr_html_, height=600)
                else:
                    generate_map()

            st.info(f'Figura 1 - Mapa Colorido Baseado na Variação de Valores da Variável Alvo.')
        
        st.divider()
        quebra_pagina()

        st.subheader('Análise Estatística')

        dfmc = df.groupby(df.columns[0])[df.columns[-1]].apply(lambda x: x.mode().iloc[0]).reset_index()
        dfm = df.groupby(df.columns[0])[df.columns[3:]].apply(lambda x: x.mode().iloc[0]).reset_index()
        dfmc[dfmc.columns[-1]] = dfmc[dfmc.columns[-1]].round(2)

        st.markdown('''A tabela de estatísticas fornece um resumo estatístico descritivo da variável alvo para os municípios analisados. Os valores apresentados 
                    incluem a contagem de observações, média, desvio padrão, valores mínimos e máximos, bem como os percentis 25%, 50% 
                    (mediana) e 75%. Estas estatísticas são úteis para entender a distribuição e a variabilidade entre os municípios.''')
        
        botao_estatisticas = st.button('Gerar tabela de estatísticas')

        if botao_estatisticas:
            st.dataframe(dfmc[dfmc.columns[-1]].describe().to_frame().T, column_config={
                'count': 'Contagem',
                'mean': 'Média',
                'std': 'Desvio Padrão',
                'min': 'Mínimo',
                '25%': '1° Quartil',
                '50%': 'Mediana',
                '75%': '3° Quartil',
                'max': 'Máximo'
            })            
            st.info(f'Tabela 1 - Estatísticas Descritivas da Variável Alvo')
        
        st.divider()

        st.subheader('Gráfico de Dispersão')
        st.markdown('''O gráfico de dispersão faz parte de uma análise estatística mais ampla apresentada no relatório, que visa 
                        explorar a variabilidade e o desempenho geral dos municípios. Ele permite identificar quais municípios
                        apresentam desempenhos extremos, tanto positivos quanto negativos, e como os valores da nossa variável alvo estão dispersos
                        em relação à media. Esta visualização facilita uma identificação mais superficial das áreas que necessitam de maior atenção e recursos.''')
        
        botao_grafico_dispersao = st.button('Gerar gráfico de dispersão')

        if botao_grafico_dispersao:
            variavel = df.columns[-2]
            variavel = st.selectbox('Selecione a variável', df.columns[3:-1])
            nome_variavel_padrao = df.columns[-2]
            st.markdown(f'Caso queira trocar a variável padrão, que é "{nome_variavel_padrao}", selecione uma nova variável e gere o gráfico de dispersão novamente.')
            # Create a scatterplot of the penultimate column
            fig = px.scatter(
                dfm.reset_index(),
                y=variavel,
                x=dfmc.columns[0],
                # size=dfmc.columns[-1],
                hover_name="Município",
                color=variavel,
                color_continuous_scale='icefire_r',
            )

            # Show the scatterplot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            st. info(f'Gráfico 1 - Gráfico de Dispersão da Distribuição da Variável Selecionada por Município')

        st.divider()    

        st.subheader('Arvore de Decisão')

        st.markdown(''' Esta seção divide-se em duas partes: Primeiro, uma tabela que lista as variáveis utilizadas no modelo de árvore de decisão juntamente com sua importância relativa. Em seguida, a própria imagem da árvore de decisões. 
                   ''')
        
        st.markdown(''' A importância de uma variável indica quanto ela contribui para a decisão final do modelo. Valores mais altos de importância sugerem que a variável tem um impacto maior na previsão do modelo. Dessa forma, quanto maior o valor
                   de sua importância na tabela, maior a importância dessa variável em geral (desconsiderando agrupamentos). Da mesma forma, quanto mais alto ela estiver posicionada na Árvore de Decisão, maior sua importância.
                   Lembrando que essa Árvore de Decisão mostra a importância das variáveis num contexto mais amplo e desconsidera a análise posterior utilizando agrupamentos.
        ''')
        botao_arvore = st.button('Gerar árvore de decisão')

        if botao_arvore:
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

            st.dataframe(feature_importances, column_config={
                '': 'Variáveis',
                'importance': 'Importância'
            })

            st.info(f'Tabela 2 -  Importância das Variáveis no Modelo de Árvore de Decisão')
            


            # Create a larger figure
            fig, ax = plt.subplots(figsize=(20, 20))

            # Plot the decision tree with larger fonts
            tree.plot_tree(reg, ax=ax, feature_names=X.columns, filled=True, fontsize=10)

            # Show the plot in Streamlit
            st.pyplot(fig)

            st.info(f'Figura 2 - Árvore de Decisão')

            st.markdown('Você chegou ao fim da seção de Análise Estatística Exploratória. Siga para a seção de Análise por Grupos.')