import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from weasyprint import HTML
from PIL import Image
import base64
import os

from my_utils import add_cabecalho

html_table = ''

def relatorio_regioes():
    secao6()
    
    imagem = Image.open('cabecalho.jpeg')
    st.image(imagem, use_column_width=True)
    
    st.title('Relatório das Regiões 🗺️')

    with st.expander("Relatório", expanded=True):
        # Ler o PDF
        with open('secao6.pdf', "rb") as f:
            pdf_contents = f.read()

        # Baixar o PDF quando o botão é clicado
        b64 = base64.b64encode(pdf_contents).decode()
        st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Relatório das Regiões.pdf"><button style="background-color: #008CBA; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">Download PDF</button></a>', unsafe_allow_html=True)

        st.markdown(html_table, unsafe_allow_html=True)

def secao6():
    df = pd.read_csv('Regiões-PE.csv')
    df = df.drop(['IBGE', 'Segmento Fiscalizador', 'GRE', 'geom'], axis=1)
    global html_table
    html_table = "<table style='margin-left: auto; margin-right: auto; width: 627px; border: 2px solid grey;'>"
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
        html_table += f"<td style='border: 1px solid grey;'> {index + 1} </td>"
        html_table += f"<td style='border: 1px solid grey;'> {row['Nome Município']} </td>"
        html_table += f"<td style='border: 1px solid grey;'> {row['Mesorregião']} </td>"
        html_table += f"<td style='border: 1px solid grey;'> {row['Microrregião']} </td>"
        html_table += "</tr>"
    html_table += "</table>"

    html_pdf = f"""<!DOCTYPE html>
                        <html lang="pt-BR">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <style>
                                @media print {{
                                    @page {{
                                        margin-top: 0.75in;
                                        size: Letter;
                                    }}
                                }}
                            </style>
                        </head>
                        <body> <h1 style='font-family: "Helvetica"; text-align: center; font-weight: bold;'>Relatório das Regiões</h1>
                            {html_table} 
                        </body>
                        </html>"""
    
    # Salvar o HTML em um arquivo temporário
    filename = "temp.html"
    with open(filename, "w") as f:
        f.write(html_pdf)

    # Converter o HTML em PDF usando WeasyPrint
    pdf_filename = 'secao6.pdf'
    HTML(filename).write_pdf(pdf_filename)
    add_cabecalho(pdf_filename)

    # Remover os arquivos temporários
    os.remove(filename)
