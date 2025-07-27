# 🎯 Dashboard de Avaliação de Candidatos

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/streamlit-1.23%2B-orange)](https://streamlit.io/) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📑 Sumário

- [Sobre](#sobre)  
- [Instalação](#instalação)  
- [Uso](#uso)  
- [Estrutura do Projeto](#estrutura-do-projeto)  
- [Configuração da API (Google Gemini)](#configuração-da-api-google-gemini)  
- [Treinamento do Modelo](#treinamento-do-modelo)  
- [Contribuição](#contribuição)  
- [Licença](#licença)  

---

## Sobre

Este projeto é um **dashboard interativo** desenvolvido com [Streamlit](https://streamlit.io/) para apoiar todo o fluxo de **avaliação de candidatos**:

- **Pesquisar Vagas** por ID e visualizar detalhes brutos.  
- **Pesquisar Aplicantes** por ID e inspecionar perfil.  
- **Predição**: ranking dos melhores candidatos para uma vaga usando um modelo XGBoost treinado com embeddings de texto e SMOTE.  
- **Chat Entrevista**: chat contínuo que gera e refina perguntas de entrevista via API do **Google Gemini**.  

---

## Instalação

1. **Clone o repositório**  
   ```bash
   git clone https://github.com/seuusuario/datathon.git
   cd datathon

2. **Crie e ative um ambiente virtual**  
   ```bash
    python -m venv .venv
    # Linux / macOS
    source .venv/bin/activate
    # Windows (PowerShell)
    .\.venv\Scripts\activate

3. **Instale dependências**  
   ```bash
    pip install --upgrade pip
    pip install -r requirements.txt

## Uso

Rode o dashboard principal com:
    ```bash
    streamlit run app.py

- O Streamlit abrirá em http://localhost:8501 no seu navegador.

- Use o menu lateral “Pages” para navegar entre as funcionalidades.

## Estrutura do Projeto

    ```text
    datathon/
    ├── app.py
    ├── pages/
    │   ├── 01_Apresentacao.py
    │   ├── 02_PesquisarVagas.py
    │   ├── 03_PesquisarAplicantes.py
    │   ├── 04_Predicao.py
    │   └── 05_ChatEntrevista.py
    ├── models/
    │   ├── model.pkl
    │   └── label_encoder.pkl
    ├── data/
    │   ├── raw/             ZIPs originais com JSONs
    │   ├── flat/            CSVs flat gerados (sem limpeza)
    │   └── processed/       CSVs processados prontos para o modelo
    ├── scripts/
    │   └── process_data.py  pré-processamento e extração
    ├── utils/
    │   ├── data_loader.py
    │   └── predictor.py
    ├── assets/
    │   └── logo_dashboard.png
    ├── .streamlit/
    │   ├── config.toml
    │   └── secrets.toml
    └── requirements.txt


## Configuração da API (Google Gemini)

Para habilitar o **Chat Entrevista** usando a API do Google Gemini, siga estes passos:

1. Crie a pasta `.streamlit` na raiz do projeto (se ainda não existir).

2. Dentro dela, crie o arquivo `secrets.toml` com o conteúdo:

   ```toml
   [gemini]
   api_key = "SUA_CHAVE_API_DO_GEMINI"

## Treinamento do Modelo

Se você precisar (re)gerar os dados processados e treinar o modelo, siga os passos abaixo:

1. **Extrair e processar os JSONs**  
   ```bash
   python scripts/process_data.py \
     --raw_dir data/raw \
     --work_dir data/temp \
     --output_dir data/processed \
     --flat_dir data/flat

2. **Treinar o modelo**  
   ```bash
   python models/train_model.py

Gera / atualiza:

- data/processed/dataset_for_model.csv

- models/model.pkl

- models/label_encoder.pkl

- models/metrics.json