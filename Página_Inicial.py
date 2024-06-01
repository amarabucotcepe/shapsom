import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import globals
from som import rodar_algoritmo
import altair as alt
from my_utils import create_map, HEX_SHAPE, verificarColunaDesc, convert_numeric
import numpy as np
import statistics

from pagess.Análise_Estatística_Exploratória import pagina_analise_estatistica_exploratoria
from pagess.Análise_Por_Grupos import pagina_analise_por_grupos
from pagess.Anomalias import pagina_anomalias
from pagess.Relatório_das_Regiões import relatorio_regioes
from pagess.Relatório_dos_Municípios import relatorio_municipios


imagem = Image.open('pixelcut-export.png')
st.image(imagem, use_column_width=True)

def pagina_inicial():
    

    st.title("Relatório para Suporte às Auditorias do Tribunal de Contas do Estado de Pernambuco")
    st.subheader("Inserção de Dados e Parametrizações.")
    title = st.text_input("Informe o nome do relatório a ser gerado", help='Esse nome será utilizado no título do arquivo de PDF que será gerado ao fim da aplicação.')
    tipo = st.radio('Escolha um tipo de arquivo. Os tipos de arquivo suportados para upload são CSV e Excel.',['csv','excel'], help='CSV (Comma-Separated Values): Este é um formato de arquivo simples que usa uma vírgula para separar os valores. Excel: Este é um formato de planilha criado pela Microsoft. Os arquivos Excel podem conter dados em várias planilhas, além de permitir a inclusão de gráficos, fórmulas e outras funcionalidades avançadas. ')
    st.markdown('Atente-se a como sua planilha está organizada! Tente deixá-la no formato do modelo padrão.')

    download_file = 'modelo.csv' if tipo == 'csv' else 'modelo.xslx'

    with st.expander("Gostaria de baixar o modelo padrão de planilha?", expanded=False):
        st.download_button('Modelo', 'modelo', file_name=download_file, help='Modelo de planilha a ser enviada')

    file = st.file_uploader("Faça upload da sua planilha", type=['csv'], help='Caso sua planilha já esteja no mesmo formato do modelo (ou seja, com as colunas semelhantes), faça o upload dela. Caso contrário, faça o download da planilha modelo e preencha com seus dados.')
    
    if file:
        df = pd.read_csv(file, sep=',') if tipo == 'csv' else pd.read_excel(file)
        globals.original_database = df.copy()

        if(verificarColunaDesc(globals.original_database)):
            globals.current_database = globals.original_database.drop(globals.original_database.index[0]).applymap(convert_numeric)
            globals.current_database.index = globals.current_database.index-1
        else:
            globals.current_database = globals.original_database

        globals.current_database = globals.current_database.dropna()
        globals.current_database_name =  file.name.split(".")[0]

        numeric_cols = list(globals.current_database.select_dtypes(include=['float64', 'int64']).columns)
        textual_cols = list(globals.current_database.select_dtypes(include=['object']).columns)
        st.divider()

        st.info("Caso deseje modificar a escolha de colunas padrões, clique na opção abaixo:")
        with st.expander("Escolher colunas", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                globals.current_label_columns = st.multiselect("Nome", textual_cols, default=[textual_cols[0]], max_selections=1, help='Selecione a coluna que será usada como o identificador principal do conjunto de dados. Esta coluna geralmente contém valores únicos, como nomes de municípios. Por padrão, é a primeira coluna da sua planilha.')
            with col2:
                globals.current_input_columns = st.multiselect("Entradas", numeric_cols, default=numeric_cols[:-1], help='As colunas marcadas como "Entrada" são aquelas que contêm as variáveis independentes. Estes são os dados que serão usados para analisar o valor de saída.')
            with col3:
                globals.current_output_columns = st.multiselect("Saída", numeric_cols, default=[numeric_cols[-1]], max_selections=1, help='A coluna marcada como "Saída" contém a variável dependente ou o valor que se deseja prever ou analisar. Esta coluna representa o resultado que é influenciado pelos dados das colunas de entrada. Por padrão, deve ser a última coluna da sua planilha.')

        st.info("Caso não queira modificar as colunas selecionadas por padrão, clique no botão 'Confirmar Colunas'")
        choose_columns = st.button("Confirmar Colunas")
        if choose_columns:
            globals.som_chart = None
            globals.file_uploaded_start_flag = True

            crunched_cols = globals.current_label_columns + numeric_cols
            crunched_df = globals.current_database[crunched_cols]
            municipio_dfs = crunched_df.groupby(globals.current_label_columns[0])
            list_of_dfs = [group_df for _, group_df in municipio_dfs]
            new_df_avg = []
            new_df_std = []

            for l in list_of_dfs:
                flatlist = lambda a : np.array(a).flatten().tolist()
                calc_std = lambda a : statistics.stdev(a) if len(a) > 1 else 0
                mun_txt = [l[globals.current_label_columns[0]].values[0]]
                mun_avg = [float(np.average(flatlist(l[c].values))) for c in numeric_cols]
                mun_std = [float(calc_std(flatlist(l[c].values))) for c in numeric_cols]
                new_df_avg.append(mun_txt + mun_avg)
                new_df_std.append(mun_txt + mun_std)

            ################################################################################################
            globals.crunched_df = pd.DataFrame(new_df_avg, columns=crunched_cols) # MÉDIA DAS UBS          #
            globals.crunched_std = pd.DataFrame(new_df_std, columns=crunched_cols) # DESVIO PADRÃO DAS UBS #
            ################################################################################################

        selected_df = df.drop(columns=globals.current_hidden_columns)

        st.divider()

        if globals.file_uploaded_start_flag or globals.som_chart is None:
            globals.som_chart = alt.Chart(pd.DataFrame([], columns=["x","y","Nota"])).mark_point(filled=True, shape=HEX_SHAPE).encode(
                x=alt.X('x', scale=alt.Scale(domain=(0,30))),
                y=alt.Y('y', scale=alt.Scale(domain=(0,30))),
                size=alt.Size('Nota', scale=alt.Scale(domain=(0,1))),
                color=alt.Color("Cor:N", scale=None)
            ).interactive().configure_view(fill='black').properties(width=400, height=400, title="Mapa SOM (Aguardando dados...)")
            globals.som = st.altair_chart(globals.som_chart, use_container_width=True)
        else:
            globals.som = st.altair_chart(globals.som_chart, use_container_width=True)

        with st.expander("Caso deseje modificar o mapa SOM acima, clique aqui. Caso contrário, clique em Iniciar Análise.", expanded=False):
            st.markdown('Essa é uma opção avançada que acabará modificando a estruturação do mapa que foi gerado acima. Leia as instruções sobre cada parâmetro e ajuste conforme sua vontade.')
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
        globals.use_shap = st.checkbox("Incluir Análise Individual dos Municípios", help='Selecione para obter, ao fim da execução, uma análise completa dos municípios de sua escolha individualmente')

        submit_button = st.button('Iniciar Análise')
        
        if submit_button:
            st.dataframe(selected_df)
            rodar_algoritmo()
    else:
        globals.file_uploaded_start_flag = False
        globals.som_chart = None

tab1, tab2, tab3, tab4, tab6, tab7= st.tabs(["Página Inicial", "Análise Estatística Exploratória", "Análise Por Grupos", "Anomalias", 'Relatório de Regiões', 'Relatório de Municípios'])
with tab1:
   pagina_inicial()
with tab2:
   pagina_analise_estatistica_exploratoria()
with tab3:
    pagina_analise_por_grupos()
with tab4:
    pagina_anomalias()
with tab6:
     relatorio_regioes()
with tab7:
     relatorio_municipios()