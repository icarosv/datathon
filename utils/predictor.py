from pathlib import Path
import pandas as pd
import streamlit as st
from joblib import load

# Diretórios
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def _safe_load(path: Path):
    """Função auxiliar para verificar existência do arquivo antes de carregar."""
    if not path.exists():
        st.error(f"Arquivo de modelo não encontrado: {path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return load(path)


@st.cache_resource
def load_model(model_filename: str = "model.pkl"):
    """
    Carrega o modelo treinado salvo na pasta models/.
    """
    return _safe_load(MODELS_DIR / model_filename)


@st.cache_resource
def load_label_encoder(filename: str = "label_encoder.pkl"):
    """
    Carrega o LabelEncoder usado durante o treinamento.
    """
    return _safe_load(MODELS_DIR / filename)


def predict(features: pd.DataFrame) -> pd.Series:
    """
    Gera previsões de rótulos para um conjunto de features.
    """
    model = load_model()
    preds = model.predict(features)
    return pd.Series(preds, index=features.index)


def predict_proba(features: pd.DataFrame) -> pd.DataFrame:
    """
    Gera probabilidades das classes para um conjunto de features.
    """
    model = load_model()
    proba = model.predict_proba(features)
    classes = model.classes_
    return pd.DataFrame(
        proba,
        columns=[f"prob_{cls}" for cls in classes],
        index=features.index,
    )


def rank_applicants_for_vaga(
    vaga_id: int,
    df_features: pd.DataFrame,
    top_n: int = 10,
    positive_class: str = "positiva",
) -> pd.DataFrame:
    """
    Retorna um ranking de candidatos para a vaga especificada.
    O score é a probabilidade da classe positiva.
    """
    # 1) Filtra apenas registros da vaga
    df = df_features[df_features["id_vaga"] == vaga_id].copy()
    if df.empty:
        return df

    # 2) Armazena ids para retornar depois
    ids = df[["id_vaga", "id_applicant"]].copy()

    # 3) Remove o target, se presente
    df = df.drop(columns=["classificacao"], errors="ignore")

    # 4) Monta X excluindo colunas de id
    X = df.drop(columns=["id_vaga", "id_applicant"], errors="ignore")

    # ─────────────────────────────────────────────────────────────────────────────
    # 5) Forçar TODAS as colunas para numérico, preencher NaN com 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    # ─────────────────────────────────────────────────────────────────────────────

    # 6) Gera probabilidades
    proba_df = predict_proba(X)

    # 7) Traduz o positive_class ("positiva") pro índice numérico via o LabelEncoder
    le = load_label_encoder()
    try:
        positive_idx = le.transform([positive_class])[0]
    except Exception:
        # se o usuário passar "1" ou similar diretamente
        positive_idx = int(positive_class)

    score_col = f"prob_{positive_idx}"
    if score_col not in proba_df.columns:
        raise ValueError(
            f"Classe positiva '{positive_class}' não encontrada. "
            f"Model.classes_ = {list(le.classes_)}"
        )

    # 8) Anexa o score e ordena
    ids["score"] = proba_df[score_col].values
    return ids.sort_values("score", ascending=False).head(top_n)
