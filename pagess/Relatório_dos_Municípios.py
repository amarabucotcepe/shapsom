import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from weasyprint import HTML
from tqdm import tqdm
from PIL import Image
import globals
import zipfile
import base64
import os

from my_utils import add_cabecalho

global current_list_labels
global shape_results
global shap_columns
global cluster_dict

def relatorio_municipios():
    st.subheader('Seção 8 - Relatório Individual dos Municípios 🏙️')

    list_all_labels = [m for m in globals.shape_results.keys()]

    with st.form("form"):
        st.subheader("Selecione os Municípios")
        list_selected_labels = st.multiselect("Municípios", list_all_labels, help='Selecione os municípios para gerar os relatórios individuais de cada um deles', key="my_multiselect")
        use_mark_all = st.checkbox("Selecionar Todos", help="Selecione para marcar todos os municípios")
        st.markdown('Para usar essa funcionalidade é necessário que a opção "Incluir Análise Individual dos Municípios" esteja Ativada.')
        button = st.form_submit_button('Executar')

    if button:
        if globals.use_shap:
            globals.current_list_labels = list_all_labels if use_mark_all else list_selected_labels
            gerar_anexos()
        else:
            st.markdown('A opção "Incluir Análise Individual dos Municípios" está Desativada, por favor execute novamente com esta opção Ativa.')


def gerar_anexos():
    pdf_filenames = []
    st.write("#### Relatórios Individuais")
    progress_bar = st.progress(0)
    total_iterations = len(globals.current_list_labels) 
    for i, municipio in tqdm(enumerate(globals.current_list_labels), desc="Gerando Anexos", total=total_iterations, unit="it/s"):
        progress_bar.progress(int((i + 1) / total_iterations * 100))
        with st.expander(municipio, expanded=False):
            dados = {}
            for cluster in globals.cluster_dict.keys():
                for coord in globals.cluster_dict[cluster].keys():
                    if municipio in globals.cluster_dict[cluster][coord]['labels']:
                        dados['cluster'] = globals.cluster_dict[cluster][coord]['cluster']
                        dados['feature'] = list(globals.cluster_dict[cluster][coord]['cluster_scores'].keys())[-1]
                        dados['cluster_score'] = globals.cluster_dict[cluster][coord]['cluster_scores'][dados['feature']]
                        dados['labels'] = globals.cluster_dict[cluster][coord]['labels']
                        dados['scores'] = globals.cluster_dict[cluster][coord]['score'][dados['feature']]

            table1 =  f"<h1 style='font-family: \"Helvetica\"; text-align: center; font-weight: bold;'>{municipio}</h1>"
            table1 +=  '<table style="margin-left: auto; margin-right: auto; margin-top: 40px; width: 627px; border: 2px solid grey;">'
            table1 += f"<caption style='color: #8B4513; caption-side: top; border: 2px solid grey; text-align: center; font-weight: bold; font-size: 23px; padding: 10px 0; '> Resultado da Influência dos Fatores na {dados['feature']} </caption>"
            table1 +=  '<thead>'
            table1 +=  "<tr style='font-size: 18px; padding: 9px; text-align: center;'>"
            table1 +=  '<th style="border: 1px solid grey;"> Índice </th>'
            table1 +=  '<th style="border: 1px solid grey;"> Fatores </th>'
            table1 +=  '<th style="border: 1px solid grey;"> Valor </th>'
            table1 +=  '<th style="border: 1px solid grey;"> Influência </th>'
            table1 +=  '</tr>'
            table1 +=  '</thead>'

            for linha in range(len(globals.shap_columns)):
                data  = globals.shape_results[municipio]['data'][linha]
                value = globals.shape_results[municipio]['values'][linha]
                table1 +=  '<tr style="text-align: center;">'
                table1 += f'<td style="border: 1px solid grey;"> {linha + 1} </td>'
                table1 += f'<td style="border: 1px solid grey;"> {globals.shap_columns[linha]} </td>'
                table1 += f'<td style="border: 1px solid grey;"> {data:0.2f} </td>' if data != 0 else '<td style="border: 1px solid grey;"> 0 </td>'
                table1 += f'<td style="color: blue; border: 1px solid grey;">' if value > 0 else (f'<td style="color: red; border: 1px solid grey;">' if value < 0 else f'<td style="border: 1px solid grey;">')
                table1 += f'{value:0.3f}' if value != 0 else '0'
                table1 += '</td>'
                table1 += '</tr>'
            table1 +=  '<tr>'
            table1 +=  '<td colspan="4" style="font-size:15px; text-align:left;">'
            table1 += f"<span style=\"color:blue;\">&#x25A0;</span> : INFLUÊNCIA <span style=\"color:blue;\">POSITIVA</span> (AUMENTA A {dados['feature'].upper()}).<br>"
            table1 += f"<span style=\"color:red;\">&#x25A0;</span> : INFLUÊNCIA <span style=\"color:red;\">NEGATIVA</span> (DIMINUI A {dados['feature'].upper()})."
            table1 +=  '</td>'
            table1 +=  '</tr>'
            table1 +=  '</table>'

            table2 = '<table style="margin-left: auto; margin-right: auto; margin-top: 60px; width: 627px; border: 2px solid grey;">'
            table2 += '<caption style="color: #8B4513; caption-side: top; border: 2px solid grey; text-align: center; font-weight: bold; font-size: 23px; padding: 10px 0;"> Fatores que Mais Influenciaram </caption>'
            table2 += '<thead>'
            table2 += "<tr style='font-size: 18px; padding: 9px; text-align: center;'>"
            table2 += '<th style="width:50%; border: 1px solid grey;" colspan="2"> Positivamente </th>'
            table2 += '<th style="width:50%; border: 1px solid grey;" colspan="2"> Negativamente </th>'
            table2 += '</tr>'
            table2 += '</thead>'

            numero_atributos = 3
            values = globals.shape_results[f'{municipio}']['values']
            maiores_valores = [v for v in sorted(values, reverse=True)[:numero_atributos] if v > 0]
            menores_valores = [v for v in sorted(values)[:numero_atributos] if v < 0]
            indices_maiores_valores = {}
            indices_menores_valores = {}
            for i, v in enumerate(values):
                if v in maiores_valores and i not in indices_maiores_valores:
                    indices_maiores_valores[i] = v
                elif v in menores_valores and i not in indices_menores_valores:
                    indices_menores_valores[i] = v
            indices_maiores_valores = dict(sorted(indices_maiores_valores.items(), key=lambda item: item[1]))
            indices_menores_valores = dict(sorted(indices_menores_valores.items(), key=lambda item: item[1], reverse=True))

            for linha in range(numero_atributos):
                chave_maior, valor_maior = indices_maiores_valores.popitem() if indices_maiores_valores else (None, None)
                chave_menor, valor_menor = indices_menores_valores.popitem() if indices_menores_valores else (None, None)
                table2 += '<tr style="font-size: 17px; text-align: center;">'
                table2 += f'<td style="width:40%; border: 1px solid grey;"> {globals.shap_columns[chave_maior] if chave_maior is not None else "---"} </td>'
                table2 += f'<td style="color: blue; width:10%; border: 1px solid grey;">' + (f'{valor_maior:.3f}' if valor_maior is not None else "---") + '</td>'
                table2 += f'<td style="width:40%; border: 1px solid grey;"> {globals.shap_columns[chave_menor] if chave_menor is not None else "---"} </td>'
                table2 += f'<td style="color: red; width:10%; border: 1px solid grey;">' + (f'{valor_menor:.3f}' if valor_menor is not None else "---") + '</td>'
                table2 += '</tr>'
            table2 += '</table>'

            score_municipio = dados['scores'][dados['labels'].index(municipio)]
            table3 =   '<table style="margin-left: auto; margin-right: auto; margin-top: 60px; width: 627px; border: 2px solid grey;">'
            table3 += f"<caption style='color: #8B4513; caption-side: top; border: 2px solid grey; text-align: center; font-weight: bold; font-size: 23px; padding: 10px 0; '> {dados['feature']} </caption>"
            table3 += '<thead>'
            table3 +=  "<tr style='font-size: 18px; padding: 9px; text-align: center;'>"
            table3 += f"<th style='border: 1px solid grey;'> Grupo {dados['cluster'] + 1}</th>"
            table3 +=  '<th style="border: 1px solid grey;"> Município </th>'
            table3 +=  '</tr>'
            table3 += '</thead>'
            table3 +=  '<tr style="text-align: center; font-size: 17px;">'
            table3 += f"<td style='border: 1px solid grey;'> {dados['cluster_score']*100:0.2f} % </td>" if dados['cluster_score'] != 0 else '<td style="font-size: 17px; border: 1px solid grey;"> 0 % </td>'
            table3 += f'<td style="color: blue; border: 1px solid grey;">' if score_municipio > dados['cluster_score'] else (f'<td style="color: red; font-size: 17px; border: 1px solid grey;">' if score_municipio < dados['cluster_score'] else f'<td style="font-size: 17px; border: 1px solid grey;">')
            table3 += f'{score_municipio*100:0.2f} %' if score_municipio != 0 else '0 %'
            table3 += '</td>'
            table3 += '</tr>'
            table3 +=  '<tr>'
            table3 +=  '<td colspan="2" style="font-size:15px; text-align:left;">'
            table3 += f"<span style=\"color:blue;\">&#x25A0;</span> : VALOR <span style=\"color:blue;\">ACIMA</span> DA MÉDIA DO GRUPO.<br>"
            table3 += f"<span style=\"color:red;\">&#x25A0;</span> : VALOR <span style=\"color:red;\">ABAIXO</span> DA MÉDIA DO GRUPO."
            table3 +=  '</td>'
            table3 +=  '</tr>'
            table3 += '</table>'

            cell_labels = [label for label in dados['labels'] if label != municipio]
            cell_labels.sort()
            table4 =   '<table style="margin-left: auto; margin-right: auto; margin-top: 60px; width: 627px; border: 2px solid grey;">'
            table4 +=  '<caption style="color: #8B4513; caption-side: top; border: 2px solid grey; text-align: center; font-weight: bold; font-size: 23px; padding: 10px 0; "> Vizinhos Mais Próximos </caption>'
            table4 +=  '<thead>'
            table4 +=  "<tr style='font-size: 18px; padding: 9px; text-align: center;'>"
            table4 +=  '<th style="border: 1px solid grey;"> Nome </th>'
            table4 += f"<th style='border: 1px solid grey;'> {dados['feature']} </th>"
            table4 +=  '</tr>'
            table4 +=  '</thead>'

            for linha in range(len(cell_labels)):
                score_label = dados['scores'][dados['labels'].index(cell_labels[linha])]
                table4 +=  '<tr style="text-align: center; font-size: 17px;">'
                table4 += f'<td style="border: 1px solid grey;"> {cell_labels[linha]} </td>'
                table4 +=  '<td style="border: 1px solid grey;">'
                table4 += f'{score_label*100:0.2f} %' if score_label != 0 else '0 %'
                table4 +=  '</td>'
                table4 +=  '</tr>'
            table4 +=   '<tr>'
            table4 +=   ' <td colspan="2" style="font-size:15px; text-align: center;"> OBS: A <i>PROXIMIDADE</i> ENVOLVE O <span style="color:#00FFFF;">CONJUNTO TOTAL</span> DOS FATORES E SUAS SEMELHANÇAS, AO INVÉS DE QUESTÕES GEOGRÁFICAS. </td>'
            table4 +=   '</tr>'
            table4 +=   '</table>'

            html = f"""{table1}
                       {table2}
                       {table3}
                       {table4}"""
            
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
                            <body> {html} </body>
                            </html>"""

            # Salvar o HTML em um arquivo temporário
            filename = "temp.html"
            with open(filename, "w") as f:
                f.write(html_pdf)

            # Converter o HTML em PDF usando WeasyPrint
            pdf_filename = f'{municipio}.pdf'
            HTML(filename).write_pdf(pdf_filename)
            add_cabecalho(pdf_filename)

            pdf_filenames.append(pdf_filename)

            # Ler o PDF
            with open(pdf_filename, "rb") as f:
                pdf_contents = f.read()

            # Baixar o PDF quando o botão é clicado
            b64 = base64.b64encode(pdf_contents).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="{municipio}.pdf"><button style="background-color: #008CBA; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">Download PDF</button></a>', unsafe_allow_html=True)

            st.markdown(html, unsafe_allow_html=True)

            # Remover os arquivos temporários
            os.remove(filename)

    # Criar um arquivo zip
    zip_filename = 'Anexos.zip'
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        # Adicionar cada arquivo PDF ao arquivo zip
        for pdf_file in pdf_filenames:
            zipf.write(pdf_file)

    # Baixar o arquivo zip quando o botão é clicado
    with open(zip_filename, "rb") as f:
        zip_contents = f.read()
    b64 = base64.b64encode(zip_contents).decode()
    st.sidebar.write('#### Anexos')
    st.sidebar.markdown(f'<a href="data:application/zip;base64,{b64}" download="Anexos.zip"><button style="background-color: #008CBA; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">Download ZIP</button></a>', unsafe_allow_html=True)

    # Remove a pasta e o arquivo zip
    for pdf_file in pdf_filenames:
        os.remove(pdf_file)
    os.remove(zip_filename)
