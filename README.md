# shapsom
Clusterização de dados usando mapas auto-organizáveis e shap


## Pré-requisitos
* Docker

## Instalação

1. git clone https://github.com/amarabucotcepe/shapsom

2. docker build -t shapsom .

3. docker run -it --name shapsom -p 8501:8501 shapsom (modo produção)
3. docker run -it --name shapsom -v "$PWD":/usr/src/app -w /usr/src/app -p 8501:8501 shapsom (modo desenvolvimento)

## Estrutura

* home: ui para carregar dados e parametrizar a geração do relatório
* page/relatorio: ui para carregar relatorio, visualizar e gerar pdf
* som: classes e funções para executar mapa som
* shap: classes e funções para executar shap
* report: classes e funções para gerar relatorio