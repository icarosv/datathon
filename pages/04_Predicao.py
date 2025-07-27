import streamlit as st
import pandas as pd
from pathlib import Path

from utils.predictor import rank_applicants_for_vaga
from utils.data_loader import load_processed_dataset

# Configuração inicial
st.set_page_config(page_title="Predição de Ranking", layout="wide")

st.title("📊 Predição: Ranking de Candidatos para uma Vaga")

# Caminho para o dataset consolidado
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dataset_file = PROJECT_ROOT / "data" / "processed" / "dataset_for_model.csv"

# Carregar dataset para seleção
try:
    df_features = load_processed_dataset("dataset_for_model.csv")
except Exception as e:
    st.error(f"Erro ao carregar dataset: {e}")
    st.stop()

# Checar se as colunas necessárias existem
required_cols = {"id_vaga", "id_applicant"}
if not required_cols.issubset(df_features.columns):
    st.error("O dataset não contém as colunas obrigatórias: 'id_vaga' e 'id_applicant'.")
    st.stop()

# Seleção do ID da vaga
vaga_ids = df_features["id_vaga"].unique()
vaga_id = st.selectbox("Selecione o ID da Vaga", sorted(vaga_ids))

# Botão para gerar ranking
if st.button("Gerar Ranking"):
    with st.spinner("Gerando ranking..."):
        ranking = rank_applicants_for_vaga(vaga_id, df_features, top_n=10, positive_class="positiva")

    if ranking.empty:
        st.warning("Nenhum candidato encontrado para essa vaga.")
    else:
        st.success(f"Ranking gerado com sucesso para a vaga {vaga_id}!")
        st.dataframe(ranking, use_container_width=True)
