import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import globals
from som import rodar_algoritmo
import altair as alt
from my_utils import create_map, HEX_SHAPE
import numpy as np


st.title("RELATÓRIO PARA SUPORTE ÀS AUDITORIAS DO TRIBUNAL DE CONTAS DO ESTADO DE PERNAMBUCO ")
st.subheader("Análise de agrupamento de dados")
title = st.text_input("Título do relatório", help='Escolha o título do relatório')
tipo = st.radio('Tipo de arquivo',['csv','excel'], help='Escolha o tipo de arquivo. csv: separado por vírgula, excel: planilha excel')

download_file = 'modelo.csv' if tipo == 'csv' else 'modelo.xslx'

with st.expander("Precisa do modelo?", expanded=False):
    st.download_button('Modelo', 'modelo', file_name=download_file, help='Modelo de planilha a ser enviada')

file = st.file_uploader("Faça upload do seu arquivo", type=['csv'], help='Se já preencheu os dados na planilha modelo faça upload de um arquivo csv ou excel, ou faça o download do modelo e preencha com seus dados')

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
            return np.int64(float(x))
        else:
            return np.float64(x)
    except (ValueError,AttributeError):
        return x
    
if file is not None:
    df = pd.read_csv(file, sep=',') if tipo == 'csv' else pd.read_excel(file)
    globals.original_database = df.copy()

    if(verificarColunaDesc(globals.original_database)):
        globals.current_database = globals.original_database.drop(globals.original_database.index[0]).applymap(convert_numeric)
        globals.current_database.index = globals.current_database.index-1
    else:
        globals.current_database = globals.original_database

    globals.current_database = globals.current_database.dropna()
    globals.current_database_name = file.name.split(".")[0]
    string_list = df.columns.tolist()
    st.divider()

    st.info("Caso deseje modificar a escolha de colunas padrões, clique na opção abaixo:")
    with st.expander("Escolher colunas", expanded=False):

        col1, col2, col3, col4 = st.columns(4)
        default_col1 = [string_list[0]] if string_list else []
        with col1:
            others = []
            globals.current_label_columns = st.multiselect("Nome", [col for col in string_list if col not in others], default=default_col1, max_selections=1)

        default_col2 = string_list[1:-1] if len(string_list) > 2 else []
        with col2:
            others = globals.current_label_columns
            globals.current_input_columns = st.multiselect("Entradas", [col for col in string_list if col not in others], default=default_col2)
        
        default_col3 = [string_list[-1]] if string_list else []
        with col3:
            others = globals.current_label_columns + globals.current_input_columns
            globals.current_output_columns = st.multiselect("Saída", [col for col in string_list if col not in others],default=default_col3, max_selections=1)
        
        with col4:
            others = globals.current_label_columns + globals.current_input_columns + globals.current_output_columns
            globals.current_hidden_columns = st.multiselect("Ocultar", [col for col in string_list if col not in others])

    st.info("Caso não queira modificar as colunas selecionadas por padrão, clique no botão 'Pronto'")
    choose_columns = st.button("Pronto")
    if choose_columns:
        globals.som_chart = None
        globals.file_uploaded_start_flag = True

        numeric_cols = list(globals.current_database.select_dtypes(include=['float64', 'int64']).columns)
        crunched_cols = globals.current_label_columns + numeric_cols
        crunched_df = globals.current_database[crunched_cols]
        municipio_dfs = crunched_df.groupby(globals.current_label_columns[0])
        list_of_dfs = [group_df for _, group_df in municipio_dfs]
        new_df = []
        for l in list_of_dfs:
            new_data = [l[globals.current_label_columns[0]].values[0]] + np.array(np.average(l[numeric_cols].values, axis=0)).astype(np.float64).flatten().tolist()
            new_df.append(new_data)
        ##################################################################################################################################
        globals.crunched_df = pd.DataFrame(new_df, columns=crunched_cols) # OU ISSO AQUI É A BASE QUE VAI PRO SOM TIRANDO MÉDIA DE UBS!!!#
        ##################################################################################################################################

    selected_df = df.drop(columns=globals.current_hidden_columns)

    st.divider()

    if globals.file_uploaded_start_flag or globals.som_chart is None:
        globals.som_chart = alt.Chart(pd.DataFrame([], columns=["x","y","Nota"])).mark_point(filled=True, shape=HEX_SHAPE).encode(
            x=alt.X('x', scale=alt.Scale(domain=(0,30))),
            y=alt.Y('y', scale=alt.Scale(domain=(0,30))),
            size=alt.Size('Nota', scale=alt.Scale(domain=(0,1))),
            color=alt.Color("Cor:N", scale=None)
        ).interactive().configure_view(fill='black').properties(width=400,height=400, title="Mapa SOM (Aguardando dados...)")
        globals.som = st.altair_chart(globals.som_chart, use_container_width=True)
    else:
        globals.som = st.altair_chart(globals.som_chart, use_container_width=True)

    with st.expander("Parâmetros SOM", expanded=False):
        globals.sigma = st.slider("Sigma", min_value=1, max_value=10, value=9, help="A largura da vizinhança inicial no mapa SOM. Controla a extensão das alterações que ocorrem durante o treinamento. Um valor alto significa que mais neurônios serão influenciados durante o treinamento inicial, enquanto um valor baixo resultará em um ajuste mais fino.")
        globals.size = st.slider("Tamanho do mapa", min_value=5, max_value=50, value=30, help="O tamanho do mapa SOM, especificado pelo número total de neurônios (unidades). Mapas maiores podem representar características complexas com maior precisão, mas também requerem mais tempo de treinamento.")
        globals.lr = st.slider("Taxa de aprendizado", min_value=-5.0, max_value=-1.0, value=-3.0, step=0.25, help="Taxa de aprendizado inicial. Controla a velocidade de adaptação do mapa durante o treinamento. Valores muito altos podem levar a uma convergência instável, enquanto valores muito baixos podem resultar em um treinamento lento.")
        globals.epochs = st.slider("Épocas", min_value=100, max_value=30000, step=100, value=10000, help="Número de épocas (iterações) de treinamento. O número de vezes que o mapa será treinado em relação aos dados de entrada. Mais épocas geralmente resultam em um mapa mais bem ajustado, mas também aumentam o tempo de treinamento.")
        globals.cluster_distance = st.slider("Distância dos agrupamentos", min_value=0.5, max_value=3.0, step=0.25, value=1.5, help="A distância mínima entre agrupamentos de neurônios para considerar a formação de grupos distintos. Valores mais altos podem resultar em agrupamentos mais distintos, enquanto valores mais baixos podem mesclar grupos semelhantes.")
        globals.topology = st.radio("Topologia", options=["Retangular", "Hexagonal"], index=1, help="Topologia do mapa SOM para formação de vizinhanças.")
        globals.output_influences = st.radio("Coluna de saída influencia nos resultados (experimental)", options=["Sim", "Não"], index=0, help="Se a coluna de saída dos dados de entrada influencia nos resultados finais. Selecione 'Sim' para permitir que a coluna de saída tenha impacto na organização do mapa, ou 'Não' para desconsiderar a coluna de saída durante o treinamento.")
        update_map = st.button("Alterar parâmetros")

    has_enough_data = globals.current_label_columns and globals.current_output_columns and len(globals.current_input_columns) >= 2
    if (update_map or globals.file_uploaded_start_flag) and has_enough_data:
        globals.file_uploaded_start_flag = False
        som_iter = create_map(
            globals.crunched_df,
            cluster_distance=globals.cluster_distance,
            lr=10**globals.lr,
            epochs=globals.epochs,
            size=globals.size,
            sigma=globals.sigma,
            label_column=globals.current_label_columns[0],
            output_column=globals.current_output_columns[0],
            variable_columns=globals.current_input_columns,
            interval_epochs=globals.epochs//20,
            output_influences=globals.output_influences == "Sim",
            topology="rectangular" if globals.topology == "Retangular" else "hexagonal"
        )

        for i,som_data in enumerate(som_iter):
            load_percentage = round((i+1) * (globals.epochs//20) * 100 / globals.epochs)
            chart_title = f"Mapa SOM ({load_percentage}%)" if load_percentage < 100 else "Mapa SOM"
            chart_data = som_data
            chart_data.columns = ["Municípios", "Nota", "x", "y", "Cor", "Cor Central", "Grupo"]
            ###################################################################################
            globals.som_data = chart_data # OU USA ISSO AQUI PRA PEGAR OS RESULTADOS DO SOM!!!#
            ###################################################################################

            if globals.topology == "Retangular":
                c = alt.Chart(globals.som_data).mark_square(filled=True).encode(
                    x="x",
                    y="y",
                    size="Nota",
                    color=alt.Color("Cor:N", scale=None),
                    tooltip=["Grupo", "Nota", "Municípios"]
                ).interactive().configure_view(fill='black').properties(width=400, height=400, title=chart_title)
            else:
                c = alt.Chart(globals.som_data).mark_point(filled=True, shape=HEX_SHAPE).encode(
                    x="x",
                    y="y",
                    size="Nota",
                    color=alt.Color("Cor:N", scale=None),
                    tooltip=["Grupo", "Nota", "Municípios"]
                ).interactive().configure_view(fill='black').properties(width=400, height=400, title=chart_title)

            globals.som_chart = c
            globals.som.altair_chart(globals.som_chart, use_container_width=True)
    
    global use_shap
    globals.use_shap = st.checkbox("Criar SHAP",help='Selecione para obter análise completa dos dados')

    submit_button = st.button('Executar')
    
    if submit_button:
        st.dataframe(selected_df)
        rodar_algoritmo()
else:
    globals.file_uploaded_start_flag = False
    globals.som_chart = None
