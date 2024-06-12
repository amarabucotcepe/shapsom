import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from weasyprint import HTML
from PIL import Image
import base64
import os
import globals

from my_utils import add_cabecalho

df = None

def relatorio_regioes():
    st.title("**Sistema de Apoio a Auditorias do Tribunal de Contas do Estado 📊**")

    secao6()
    st.subheader('Seção 7 - Identificação de Mesorregiões e Microrregiões')

    st.markdown('Essa seção traz uma tabela com todos os municípios de Pernambuco, identificando suas mesorregiões e microrregiões e dando um índice para elas, que é o índice utilizado nos Mapas de Calor.')

    with st.expander("Relatório", expanded=False):
      
        st.dataframe(df, use_container_width=True)
    # globals.table_list.append('table7x1')
    st.info(f"**Tabela 7 - Municípios e Suas Mesorregiões e Microrregiões**")


def secao6():
    global df
    df = pd.read_csv('Regiões-PE.csv')
    df = df.drop(['IBGE', 'Segmento Fiscalizador', 'GRE', 'geom'], axis=1)
    df.index = range(1, len(df) + 1)
    html_table = "<table style='margin-left: -14px; margin-right: auto; width: 680px; border: 2px solid grey;'>"
    html_table += "<caption style='color: #8B4513; caption-side: top; border: 2px solid grey; text-align: center; font-weight: bold; font-size: 23px; padding: 10px 0; '>Mesorregião e Microrregião dos Municípios</caption>"
    html_table += "<thead>"
    html_table += "<tr style='font-size: 18px; padding: 9px;'>"
    html_table += "<th style='border: 1px solid grey;'> Índice </th>"
    html_table += "<th style='border: 1px solid grey;'> Nome Município </th>"
    html_table += "<th style='border: 1px solid grey;'> Mesorregião	 </th>"
    html_table += "<th style='border: 1px solid grey;'> Microrregião </th>"
    html_table += "</tr>"
    html_table += "</thead>"

    for index, row in df.iterrows():
        html_table += "<tr>"
        html_table += f"<td style='border: 1px solid grey;'> {index} </td>"
        html_table += f"<td style='border: 1px solid grey;'> {row['Nome Município']} </td>"
        html_table += f"<td style='border: 1px solid grey;'> {row['Mesorregião']} </td>"
        html_table += f"<td style='border: 1px solid grey;'> {row['Microrregião']} </td>"
        html_table += "</tr>"
    html_table += "</table>"
    html_table += f'<p class="legenda-tabela"> Tabela 7 - Municípios e Suas Mesorregiões e Microrregiões </p>'

    html_pdf = f"""<!DOCTYPE html>
                        <html lang="pt-BR">
                        <head>
                            <meta charset="latin-1">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <style>
                                @media print {{
                                    @page {{
                                        margin-top: 0.75in;
                                        size: Letter;
                                    }}
                                }}
                                .legenda-tabela {{
                                    font-family: "Arial"
                                    font-size: 20px;
                                    font-style: italic;
                                    color: blue;
                                }}
                            </style>
                        </head>
                        <body> <h1 style='font-family: "Helvetica"; text-align: center; font-weight: bold;'>Seção 7 - Identificação de Mesorregiões e Microrregiões</h1>
                            {html_table} 
                        </body>
                        </html>"""
    
    # Salvar o HTML em um arquivo temporário
    filename = "temp.html"
    with open(filename, "w") as f:
        f.write(html_pdf)
        
    # Converter o HTML em PDF usando WeasyPrint
    pdf_filename = 'secao7.pdf'
    HTML(filename).write_pdf(pdf_filename)
    add_cabecalho(pdf_filename)

    # Remover os arquivos temporários
    os.remove(filename)



   