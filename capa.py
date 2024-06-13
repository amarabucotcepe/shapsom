from weasyprint import HTML
import os
import weasyprint
import locale
from datetime import datetime

def criar_capa(tipo):
    # Configurar localidade para data
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    data_atual = datetime.now()
    mes_atual = data_atual.strftime('%B')
    ano_atual = data_atual.year

    # Template HTML com placeholders
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    font-size: 24px;
                    text-transform: uppercase;
                    font-weight: bold;
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    text-align: center;
                }}
                h3 {{
                    color: #6495ED;
                }}

                p{{
                    color: #808080
                }}
                .spacer {{
                    height: 20px;
                }}
            </style>
        </head>
        <body>
            {spacers_top}
            <p>Relatório de Apoio a Auditorias</p>
            <p>{tipo}</p>
            {spacers_bot}
            <h3>Pernambuco {mes_atual} {ano_atual}</h3>
        </body>
    </html>
    """

    # Gerar divs de espaçamento
    spacers_top ="<div class='spacer'></div>" * 22
    spacers_bot = "<div class='spacer'></div>" * 11

    # Preencher template HTML
    html_content = html_template.format(
        spacers_top=spacers_top,
        spacers_bot=spacers_bot,
        tipo=tipo,
        mes_atual=mes_atual,
        ano_atual=ano_atual
    )

    path = os.path.join(f"capa.pdf")
    weasyprint.HTML(string=html_content).write_pdf(path)
