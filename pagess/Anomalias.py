from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
from PIL import Image
import branca.colormap as cm
from branca.colormap import linear

import folium
import json
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import globals
import math
import geopandas as gpd

import plotly.graph_objects as go

# Set page configuration
#st.set_page_config(layout='wide')

def pagina_anomalias(df_SHAP_grupos_normalizado):
    st.subheader('Seção 6 - Anomalias')
    st.write("""A análise de anomalias foi conduzida utilizando um Mapa Auto-Organizável (SOM) para identificar pontos de dados que se desviam significativamente do padrão observado.
    Com as coordenadas dos pontos no SOM, o centroide do mapa foi calculado. Este centroide é determinado utilizando a mediana das coordenadas x e y de todos os pontos, o que fornece uma medida menos sensível a outliers em comparação com a média. Então, são calculadas as distâncias dos pontos para o centroide do mapa.
    Pontos que apresentaram distâncias significativamente maiores em relação ao centroide foram identificados como anômalos. Estes pontos fora do cluster principal sugerem comportamentos ou características discrepantes dos dados normais, destacando-se por estarem afastados do padrão usual.""")

    has_databases = True
    try:
        has_databases = has_databases and globals.som_chart is not None
        has_databases = has_databases and globals.som is not None
        has_databases = has_databases and globals.som_data is not None
    except:
        has_databases = False
    
    st.divider()
    try:
        if not has_databases:
            st.write("Nenhum dataset foi carregado. Por favor, carregue um dataset e tente novamente.")
        else:
            globals.som = st.altair_chart(globals.som_chart, use_container_width=True)
            df = globals.som_data
            med_x = df['x'].median()
            med_y = df['y'].median()
            st.write(f"O centroide do está localizado em: x = {med_x} e y = {med_y}")
            globals.porcentagem = st.slider("Selecione a porcentagem de anômalos esperados", min_value=1, max_value=100, step=1, value=10, help="Porcentagem de anomalias esperadas. Por exemplo, se o valor for 10, o algoritmo irá mostrar 10% dos dados como anomalias.")
            get_anomalies = st.button("Obter Anomalias")
            if get_anomalies:
                porcentagem=10
                distancias = {}
                with st.spinner('Calculando distâncias...'):
                    for i in range(df.shape[0]):
                        x = df['x'][i]
                        y = df['y'][i]
                        dist = math.sqrt(math.pow(x-med_x,2)+math.pow(y-med_y,2))
                        distancias[i] = dist
                    dist_df = pd.DataFrame({'Distância do centroide':pd.Series(distancias)})
                    df_aux = dist_df.join(df)
                   
                   
                    df_aux['Valor mais influente no grupo'] = None
                    df_aux['Valor menos influente no grupo'] = None

                    for idx, row in df_aux.iterrows():
                        grupo = row['Grupo']
                        grupo_column = f'Grupo {grupo}'

                        if grupo_column in df_SHAP_grupos_normalizado.columns:
                            # Get the column values for the specific group
                            grupo_values = df_SHAP_grupos_normalizado[grupo_column]
                            
                            # Find the max and min values
                            max_value = grupo_values.max()
                            min_value = grupo_values.min()
                            
                            # Get the names of the variables for these max and min values
                            max_variable = df_SHAP_grupos_normalizado[df_SHAP_grupos_normalizado[grupo_column] == max_value]['Nome Variável'].values[0]
                            min_variable = df_SHAP_grupos_normalizado[df_SHAP_grupos_normalizado[grupo_column] == min_value]['Nome Variável'].values[0]

                            # Assign these values to the respective columns in df_aux
                            df_aux.at[idx, 'Valor mais influente no grupo'] = f"{max_variable}: {max_value:.3f}"
                            df_aux.at[idx, 'Valor menos influente no grupo'] = f"{min_variable}: {min_value:.3f}"     
                
                
                    df_aux = df_aux.sort_values(by=['Distância do centroide'], ascending=False).head(df.shape[0]//porcentagem)
                    df_aux = df_aux.drop(['Cor', 'Cor Central'],axis=1).reset_index(drop=True)
                    st.dataframe(df_aux, use_container_width=True)

                    globals.table_list.append('table6x1')
                    st.info(f"*Tabela {len(globals.table_list)} - Anomalias Encontradas*")
                    globals.df_anomalias = df_aux
    except:
        st.write("Nenhum dataset foi carregado. Por favor, carregue um dataset e tente novamente.")
        
def criar_pdf_anomalias(df_anomalias: pd.DataFrame):
        html = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
            @media print {{
            @page {{
                margin-top: 1.5in;
                size: A4;
            }}
            }}

            body {{
                font-family: "Helvetica";
                font-weight: bold;
            }}

            header {{
                text-align: left
                margin-top: 0px; /* Espaço superior */
            }}

            .table-text {{
                text-align: justify; /* Alinha o texto com justificação */
                margin-bottom: 10px; /* Margem para alinhamento com as extremidades da página */
                font-size: 12px;
            }}
            
            .legenda-tabela {{
                font-size: 10px;
                font-style: italic;
                color:blue;
            }}

            /* Define o tamanho da tabela */
            table {{
                width: 50vw; /* 50% da largura da viewport */
                height: calc(297mm / 2); /* Metade da altura de uma folha A4 */
                border: 1px solid black; /* Borda da tabela */
                border-collapse: collapse; /* Colapso das bordas da tabela */
            }}
            /* Estilo das células */
            td, th {{
                border: 1px solid black; /* Borda das células */
                padding: 4px; /* Espaçamento interno das células */
                text-align: center; /* Alinhamento do texto */
                font-size: 12px; /* Tamanho da fonte */
            }}

            .evitar-quebra-pagina {{
                page-break-inside: avoid; /* Evita quebra de página dentro do bloco */
            }}

            </style>
            </head>

            <body>
            <header>
                <h2>Anomalias</h2>
            </header>
            <p class="table-text">A análise de anomalias foi conduzida utilizando um Mapa Auto-Organizável (SOM) para identificar pontos de dados que se desviam significativamente do padrão observado.
    Com as coordenadas dos pontos no SOM, o centroide do mapa foi calculado. Este centroide é determinado utilizando a mediana das coordenadas x e y de todos os pontos, o que fornece uma medida menos sensível a outliers em comparação com a média. Então, são calculadas as distâncias dos pontos para o centroide do mapa.
    Pontos que apresentaram distâncias significativamente maiores em relação ao centroide foram identificados como anômalos. Estes pontos fora do cluster principal sugerem comportamentos ou características discrepantes dos dados normais, destacando-se por estarem afastados do padrão usual.</p>
            
            
            *-*-*-*-*
            <p class="legenda-tabela">tabela_secao_6</p>
            
            </body>
            </html>
            """
            
        df_anomalias_editado = df_anomalias.copy()
        df_anomalias_editado = df_anomalias_editado.drop(['Nota', 'x', 'y'],axis=1)
            
        html = html.replace('*-*-*-*-*', df_anomalias_editado.to_html())
        html = html.replace('tabela_secao_6', "Tabela 6.1 - Tabela de anomalias")
        path = os.path.join(f"secao6.pdf")
        weasyprint.HTML(string=html).write_pdf(path)