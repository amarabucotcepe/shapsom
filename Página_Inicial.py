import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import globals
from som import rodar_algoritmo


st.title("ShapSom 🤖")
st.subheader("Análise de agrupamento de dados")

title = st.text_input("Título do relatório", help='Escolha o título do relatório')

tipo = st.radio('Tipo de arquivo',['csv','excel'], help='Escolha o tipo de arquivo. csv: separado por vírgula, excel: planilha excel')

if tipo == 'csv':
    download_file = 'modelo.csv'
else:
    download_file = 'modelo.xslx'

with st.expander("Precisa do modelo?", expanded=False):
    st.download_button('Modelo', 'modelo', file_name=download_file, help='Modelo de planilha a ser enviada')

file = st.file_uploader("Faça upload do seu arquivo", type=['csv'], help='Se já preencheu os dados na planilha modelo faça upload de um arquivo csv ou excel, ou faça o download do modelo e preencha com seus dados')


if file is not None:
 
    if tipo == 'csv':
        #checar qual sep está sendo usado
        df = pd.read_csv(file, sep=',')
    else:
        df = pd.read_excel(file)
    # st.dataframe(df)
    
    globals.current_database = df.dropna()
    globals.current_database_name = file.name.split(".")[0]

    string_list = df.columns.tolist()

    st.divider()

    st.subheader("Selecione as colunas")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            globals.current_label_columns = st.multiselect("Nome", string_list)
        with col2:
            options = [col for col in string_list if col not in globals.current_label_columns]
            globals.current_input_columns = st.multiselect("Entradas", [col for col in string_list if col not in globals.current_label_columns])
        with col3:
            globals.current_output_columns = st.multiselect("Saída", [col for col in string_list if col not in globals.current_label_columns and col not in globals.current_input_columns])
        with col4:
            globals.current_hidden_columns = st.multiselect("Ocultar", [col for col in string_list if col not in globals.current_label_columns and col not in globals.current_input_columns and col not in globals.current_output_columns])

    selected_df = df.drop(columns=globals.current_hidden_columns)

    st.divider()

    with st.expander("Parâmetros SOM", expanded=False):
        globals.cluster_distance = st.slider("Cluster Distance", min_value=0.5, max_value=4.0, value=1.0, step=0.1, help='Distancia entre os grupos')
        globals.epochs = st.slider("Epochs", min_value=2, max_value=5, value=3, step=1)  # Changed step to int
        globals.size = st.slider("Size", min_value=5, max_value=60, value=30, step=1)  # Changed step to int
        globals.sigma = st.slider("Sigma", min_value=1, max_value=10, value=9, step=1)  # Changed step to int
        globals.lr = st.slider("Learning Rate", min_value=-3.0, max_value=-1.0, value=-2.0, step=0.1)
    
    global use_shap
    globals.use_shap = st.checkbox("Criar SHAP",help='Selecione para obter análise completa dos dados')

    submit_button = st.button('Executar')
    
    if submit_button:
        st.dataframe(selected_df)
        rodar_algoritmo()
