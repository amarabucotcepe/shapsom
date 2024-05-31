from io import StringIO
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import numpy as np
import branca.colormap as cm
from branca.colormap import linear
import re
import folium
import json
from streamlit_folium import st_folium
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import globals

import unicodedata
from datetime import datetime
import pytz
from sortedcontainers import SortedSet

import geopandas as gpd
import os

import plotly.graph_objects as go

# Set page configuration
#st.set_page_config(layout='wide')

def pagina_analise_por_grupos():
    has_databases = True
    try:
        has_databases = has_databases and globals.current_database is not None
        has_databases = has_databases and globals.crunched_df is not None
        has_databases = has_databases and globals.som_data is not None
    except:
        has_databases = False

    if has_databases:
        df =  globals.current_database
        for i in range(globals.som_data['Grupo'].max() + 1):
            filepath = f'mapa{i}.html'
            if os.path.exists(filepath):
                os.remove(filepath)
                
        def secao1():
            
            def verificarColunaDesc(database):
                nStrings = 0
                for i in range(database.shape[1]):
                    try:
                        float(np.array(database.iloc[0])[i])
                    except ValueError:
                        nStrings+=1
                if(nStrings==database.shape[1]):
                    return True
                else:
                    return False
        
            def convert_numeric(x):
                try:
                    if float(x).is_integer():
                        return int(float(x))
                    else:
                        return float(x)
                except (ValueError,AttributeError):
                    return x

            # Cálculo de data
            current_time_utc = datetime.utcnow()
            your_timezone = pytz.timezone('America/Recife')
            current_time_local = current_time_utc.replace(tzinfo=pytz.utc).astimezone(your_timezone)
            formatted_date = current_time_local.strftime("%d/%m/%Y")
        
            if(globals.current_database is not None):
                # Gerar base de dados sem descrição completa
                if(verificarColunaDesc(globals.original_database)):
                    analyzed_database = globals.original_database.drop(globals.original_database.index[0]).applymap(convert_numeric)
                    analyzed_database.index = analyzed_database.index-1
                else:
                    analyzed_database = globals.original_database

                # Gerar primeira seção
                num_rows, num_columns = analyzed_database.shape
                texto1 = 'Este relatório foi elaborado com base nos dados presentes no arquivo “'+unicodedata.normalize('NFKC', globals.current_database_name)+'” no dia '+formatted_date+'. A tabela fornecida possui '+str(num_rows)+' linhas e '+str(num_columns)+' colunas.'
                valoresFaltantes = analyzed_database.isnull().sum().sum()
                if(valoresFaltantes==0):
                    textoFaltantes = 'A tabela não possui valores faltantes.'
                elif(valoresFaltantes==1):
                    textoFaltantes = 'A tabela possui 1 valor faltante na linha '+str(1+np.where(analyzed_database.isnull())[0][0]) +' e na coluna "'+analyzed_database.columns[np.where(analyzed_database.isnull())[1][0]]+'".'
                else:
                    textoFaltantes = 'A tabela possui '+str(valoresFaltantes)+' valores faltantes'
                    if(valoresFaltantes < 6):
                        textoFaltantes += ' nas seguintes localizações: '
                        for i in range(valoresFaltantes):
                            textoFaltantes += 'linha '+str(1+np.where(analyzed_database.isnull())[0][i]) +' e coluna "'+analyzed_database.columns[np.where(analyzed_database.isnull())[1][i]]+'", '
                        textoFaltantes = textoFaltantes[:-2]+'.'
                    else:
                        textoFaltantes += '.'

                texto2 = textoFaltantes+ ' A seguir, na tabela 1, apresentamos o dicionário de dados. É importante notar que colunas com texto ou aquelas que foram ocultadas durante a criação do mapa não foram incluídas na análise.'

                # Dicionário de dados

                def checarTipo(X,tipo):
                    if(tipo=='01'):
                        for i in X:
                            if(i not in [0,1]):
                                return False
                        return True
                    elif(tipo=='0'):
                        for i in X:
                            if(i != 0):
                                return False
                        return True
                    elif(tipo=='1'):
                        for i in X:
                            if(i != 1):
                                return False
                        return True
                    else:
                        for i in X:
                            if(not isinstance(i,tipo)):
                                return False
                        return True

                def gerarArray(X):
                    array1 = []
                    for i in X:
                        array1+= [i]
                    return array1

                def detectarTipo(X):

                    tipo = ''
                    faltantes = ''

                    if(X.isnull().sum()!=0):
                        faltantes = ' com valores faltantes'
                        X = X.dropna()

                    arrX = gerarArray(X)

                    if(checarTipo(arrX,'0')):
                        tipo ='Numérico (0)'
                    elif(checarTipo(arrX,'1')):
                        tipo ='Numérico (1)'
                    elif(checarTipo(arrX,'01')):
                        tipo ='Binário (0 ou 1)'
                    elif(checarTipo(arrX,(int,float))):
                        tipo = 'Numérico'
                        unicos = [int(num) if num == int(num) else format(num,'.1f') for num in SortedSet(X)]
                        if(len(unicos)<4):
                            tipo+= ' ('
                            for i in unicos:
                                tipo+=str(i)+', '
                            tipo = tipo[:-2]
                            tipo+= ')'
                        elif(max(arrX)<=1 and min(arrX)>=0):
                            tipo += ' (entre 0 e 1)'
                    elif(checarTipo(arrX,str)):
                        tipo = 'Textual'
                    else:
                        tipo = 'Outro'

                    return tipo+faltantes

                tiposDados = []
                for i in range(analyzed_database.columns.size):
                    tiposDados+=[detectarTipo(analyzed_database.iloc[:,i])]
                nomesDados = analyzed_database.columns.tolist()

                # Descrição de dados

                descDados = []

                if(globals.original_database.shape!=analyzed_database.shape):
                    descDados = np.array(globals.original_database.iloc[0])
                
                # Número das variáveis

                numDados = []
                varNum = 1
                for i in range(len(tiposDados)):
                    if(nomesDados[i] in globals.current_label_columns):
                        numDados += ['Nome']
                    elif(nomesDados[i] in globals.current_output_columns):
                        numDados += ['Saída']
                    elif(nomesDados[i] in analyzed_database[globals.current_input_columns].select_dtypes(include='number').columns):
                        numDados += [varNum]
                        varNum += 1
                    else:
                        numDados+=['']

                def insert_newlines(text, every=40):
                    lines = []
                    while len(text) > every:
                        split_index = text.rfind(' ', 0, every)
                        if split_index == -1:
                            split_index = every
                        lines.append(text[:split_index].strip())
                        text = text[split_index:].strip()
                    lines.append(text)
                    return '\n'.join(lines)

                def dividirlinhas(data, every):
                    if(len(data)>every):
                        data = insert_newlines(data, every=every)
                    return data
                
        
                # Geração do dicionário de dados
                if(len(descDados)==0):
                    dicionarioDados = np.array([numDados,nomesDados,tiposDados]).tolist()
                    for i in range(len(dicionarioDados[1])):
                        dicionarioDados[1][i]=dividirlinhas(dicionarioDados[1][i],80)
                    for i in range(len(dicionarioDados[2])):
                        dicionarioDados[2][i]=dividirlinhas(dicionarioDados[2][i],25)
                else:
                    dicionarioDados = np.array([numDados,nomesDados,descDados,tiposDados]).tolist()
                    for i in range(len(dicionarioDados[1])):
                        dicionarioDados[1][i]=dividirlinhas(dicionarioDados[1][i],40)
                    for i in range(len(dicionarioDados[2])):
                        dicionarioDados[2][i]=dividirlinhas(dicionarioDados[2][i],40)
                    for i in range(len(dicionarioDados[3])):
                        dicionarioDados[3][i]=dividirlinhas(dicionarioDados[3][i],25)
                dicionarioDados = np.array(dicionarioDados).T.tolist()


                dicionarioDados = pd.DataFrame(dicionarioDados)
                if(len(descDados)==0):
                    dicionarioDados.columns = ['Fator','Nome da coluna','Tipo de dado']
                else:
                    dicionarioDados.columns = ['Fator','Nome da coluna','Descrição do dado','Tipo de dado']

            def gerarRetangulo(texto):
                st.markdown(
                f'<div style="background-color: #CCCCCC; border: 2px solid #808080; border-radius: 5px; padding: 10px">{texto}</div>',
                unsafe_allow_html=True
            )
                
            def gerarEspaco():
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            

            st.subheader('Seção 1 - Descrição do arquivo de entrada')
            if( globals.current_database is None):
                st.write('Escolha a base de dados.')
            else:
                st.write('#### 1.1 Dados de Treinamento')
                gerarEspaco()
                textoSOM = 'Um dicionário de dados é uma tabela que contém informações sobre os dados disponibilizados. As informações reveladas abaixo revelam o número atribuído a cada fator, sua descrição quando disponibilizada e seu tipo de dado.'
                gerarRetangulo(textoSOM)
                gerarEspaco()
                st.write(texto1)
                st.write(texto2)

                custom_css = """
                
                <style>
                thead th {
                    background-color: #CCCCCC;
                    color: white;
                }
                </style>
                """
                st.markdown(custom_css, unsafe_allow_html=True)

                if( globals.current_database is not None):
                    st.markdown(dicionarioDados.style.hide(axis="index").to_html(), unsafe_allow_html=True)

                gerarEspaco()
                st.write('#### 1.2 Parâmetros de Treinamento')
                gerarEspaco()
                textoSOM = '''Um Mapa Auto-Organizável (SOM) é uma técnica de aprendizado não supervisionado usada para visualizar e organizar dados complexos em uma representação bidimensional. Os principais parâmetros que definem um mapa SOM incluem:
                \n●	Topologia hexagonal: Define como as células do mapa influenciam suas vizinhas em um arranjo hexagonal.
                \n●	Distância de cluster: Determina como as unidades são agrupadas com base na similaridade dos dados.
                \n●	Épocas: Representam o número de vezes que o modelo passa pelos dados durante o treinamento.
                \n●	Tamanho do mapa: Define o número total de unidades no mapa.
                \n●	Sigma: O raio de influência de cada unidade durante o treinamento.
                \n●	Taxa de aprendizado: Controla a magnitude das atualizações dos pesos das unidades durante o treinamento.
                '''
                gerarRetangulo(textoSOM)
                gerarEspaco()
                st.write('Nesta seção, apresentamos os hiperparâmetros utilizados para configurar o algoritmo. Os dados mencionados no parágrafo anterior foram aplicados a um algoritmo de Mapas Auto-Organizáveis (Mapas SOM), utilizando os seguintes parâmetros:')
                param_treino = [
                    "Topologia: "+str(globals.topology),
                    f"Distância de cluster: "+str(globals.cluster_distance),
                    f"Épocas: "+str(globals.epochs),
                    f"Tamanho do mapa: "+str(globals.size),
                    f"Sigma: "+str(globals.sigma),
                    f"Taxa de aprendizado: "+str(globals.lr)    
                ]
                for item in param_treino:
                    st.write(f"- {item}")
                
                #if(globals.current_output_columns != []):
                #    gerarEspaco()
                #    st.write('#### 1.3 Parâmetros de Triagem')
                #    gerarEspaco()
                #    textoTriagem = 'A etapa de triagem realizará uma análise filtrada em relação à saída "'+globals.current_output_columns[0]+'". Os limites mínimo e máximo determinam o intervalo percentual do filtro realizado.'
                #    gerarRetangulo(textoTriagem)
                #    gerarEspaco()
                #    st.write('Os limites utilizados para a realização da triagem foram:')
                #    param_triagem = [
                #        f"Limite mínimo: {min_filter*100:.0f}%",
                #        f"Limite máximo: {max_filter*100:.0f}%"
                #    ]
                #    for item in param_triagem:
                #        st.write(f"- {item}")
                #    gerarEspaco()

        ##############################################################################
        # FUNÇÕES AUXILIARES PARA AS SEÇÕES 2 E 5
        def formatDf(df):
            municipios = df[['Município']]
            formated_df = df.drop(columns=df.columns[:3])
            nums_columns = list(range(1, len(formated_df.columns) + 1))
            formated_df.columns = nums_columns
        
            return formated_df, municipios
            
        def generate_heatmap(data, cmap):
                largura, altura = data.shape
                sns.set_theme()
                plt.figure(figsize=(altura/9.15 ,largura/7)) 
                heatmap =sns.heatmap(data,
                    annot=False,
                    cmap=cmap,
                    square=True,
                    #vmin=0, vmax=1, 
                    cbar=True, cbar_kws={'orientation': 'vertical'})
                
                return heatmap
        ###################################################################################
        def secao2():
            st.subheader('Seção 2 - Visão Geral de Dados e Heatmap')
            st.markdown('''Esta seção traz uma análise visual da base de dados, 
                        fornecendo média e desvio padrão dos fatores disponibilizados para cada um dos municípios. 
        
                        Importante:
            As linhas representam os municípios, que estão em ordem alfabética;
            As colunas representam os fatores selecionados pelo usuário na base de dados;
            A cor de cada quadrado indica a intensidade do fator naquele município.''')
        
            col1, col2 = st.columns(2)
            with col1:
                df_previa = globals.crunched_df.drop(columns=globals.crunched_df.columns[[0, 2]], axis=1)
                crunched_df, a = formatDf(globals.crunched_df)
                st.write("**Tabela 1 - Média:**")
                st.dataframe(df_previa)
                st.subheader('Heatmap 1')
                heatmap1 = generate_heatmap(crunched_df, 'viridis')
                st.pyplot(heatmap1.figure)
            
            with col2:
                df_previa = globals.crunched_std.drop(columns=globals.crunched_std.columns[[0, 2]], axis=1)
                crunched_std, a = formatDf(globals.crunched_std)
                st.write("**Tabela 2 - Desvio Padrão:**")
                st.dataframe(df_previa)
                st.subheader('Heatmap 2')
                heatmap2 = generate_heatmap(crunched_std, 'inferno')
                st.pyplot(heatmap2.figure)
            
        def secao3():
            st.subheader('Seção 3 - Análise entre grupos')
        
            st.markdown('''<div max-width: '70%'>Nesta seção, apresentamos os grupos identificados e as variáveis que mais influenciaram na formação desses grupos.
                    
            Um "agrupamento" reúne dados que são mais semelhantes em termos de suas características globais. Esses grupos são utilizados na aplicação de IA através de bases de dados (tabelas) fornecidas pela área usuária para o processamento com Redes Neurais Artificiais.  
            "Agrupamento" é o processo de reunir, por exemplo, municípios, com base em suas semelhanças, visando realizar triagens para guiar auditorias. Os mapas gerados pelas Redes SOM são topológicos, não geográficos, o que permite tratar diferentes entidades de forma justa e diligente, com base em suas semelhanças.<div>''', unsafe_allow_html=True)
            
            #st.text(globals.som_data)
            tabela_df = globals.shapsom_data.copy()
            tabela_df.drop(['Municípios', 'Nota', 'SHAP Normalizado', 'x', 'y', 'Cor', 'SHAP Original'], axis=1, inplace=True)

            
            tabela_unica = tabela_df.drop_duplicates(subset=['Cor Central', 'Grupo'])

            nome_variavel_coluna = 'Nome Variável'
            grupos_colunas = sorted(tabela_unica['Grupo'].unique())
            colunas_novo_df = [nome_variavel_coluna] + [f'Grupo {grupo}' for grupo in grupos_colunas]

            
            novo_df = pd.DataFrame(columns=colunas_novo_df)

            for idx, nome_variavel in enumerate(globals.shap_columns):
                novo_df.at[idx, 'Nome Variável'] = nome_variavel
                for grupo in grupos_colunas:
                    valores_grupo = tabela_unica.loc[tabela_unica['Grupo'] == grupo, 'SHAP Media Cluster'].values
                    if len(valores_grupo) > 0 and len(valores_grupo[0]) > idx:
                        novo_df.at[idx, f'Grupo {grupo}'] = valores_grupo[0][idx]
                    else:
                        novo_df.at[idx, f'Grupo {grupo}'] = None
            
            #print(novo_df)
            #novo_df = novo_df.style.apply_index()
            
            novo_df = novo_df.style.set_caption("Tabela de atributos vs agrupamento")
            
            st.dataframe(novo_df, hide_index=True,)


        def secao4():
            #Criando as variáveis
            original_df = globals.crunched_df
            df = globals.shapsom_data
            max_grupo = df['Grupo'].max()
            df_expandido = df.assign(municipios=df['Municípios'].str.split(',')).explode('municipios').reset_index(drop=True)
            df_expandido = df_expandido.drop(columns=['Municípios', 'x', 'y'])

            output_column = original_df.columns[-1]
            output_values = original_df.iloc[:, -1]
            df_expandido[output_column] = output_values.values

            grupos = df_expandido.groupby('Grupo')

                    
            st.subheader('Seção 4 - Diferença entre grupos')
            st.markdown('''A análise comparativa entre os agrupamentos é conduzida combinando todas as informações 
                        da "Análise de Agrupamento" (Seção 3), organizando-as em uma disposição paralela. Isso tem o 
                        objetivo de destacar de forma mais clara as disparidades nas estruturas dos agrupamentos.''')    

            for i in range(max_grupo+1):
                if i in grupos.groups:
                    #Tabelas
                    grupo_df = grupos.get_group(i)
                    media_valor = grupo_df[output_column].mean()
                    media_valor = media_valor.round(2)
                    cor_grupo = grupo_df['Cor'].iloc[0]
                    lista_cores = grupo_df['Cor'].tolist()

                    st.info(f'Grupo {i}')
                    st.text(f'Média de {output_column} do grupo: {media_valor}')

                    def apply_color(val):
                        return f"background-color: {cor_grupo}; "

                    st.dataframe(grupo_df.style.applymap(apply_color).format({output_column:'{:.2f}'}), column_order=['municipios', 'Nota', 'Grupo', 'Cor', output_column] ,column_config={
                        'municipios': 'Municípios',
                        'Nota': None,
                        'Grupo': 'Grupo',
                        'Cor': None
                        }             
                    )

                    #Mapas
                    
                    def generate_map():
                        # Convert the DataFrame to a GeoDataFrame
                        gdf = gpd.read_file('PE_Municipios_2022.zip')
                        gdf = gdf.merge(grupo_df[[grupo_df.columns[2],grupo_df.columns[-2]]], left_on='NM_MUN', right_on=grupo_df.columns[-2])

                        fig, ax = plt.subplots(1, 1)

                        custom_cmap = mcolors.ListedColormap([cor_grupo])
                        
                        values_range = np.linspace(0, 1, 10)

                        # Plot the map and apply the custom colormap
                        m = gdf.explore(column=grupo_df.columns[2], cmap=custom_cmap, vmin=0, vmax=1)

                        components.html(m._repr_html_(), height=600)

                        outfp = f"mapa{i}.html"

                        m.save(outfp)

                    with st.spinner('Gerando mapa...'):
                        if os.path.exists(f'mapa{i}.html'):
                            m_repr_html_ = open(f'mapa{i}.html').read()
                            components.html(m_repr_html_, height=600)
                        else:
                            generate_map()

        def secao5():
            st.subheader('Seção 5 - Filtro de Triagem')
            st.text('Explicar como esse filtro está sendo aplicado e falar que ele também será aplicado na parte de anomalias')

        st.title('Análise Por Grupos com SHAP/SOM')
        for i,secao in enumerate([secao1, secao2, secao3, secao4, secao5]):
            try:
                secao()
            except:
                st.subheader(f'Seção {i+1} - Erro')
