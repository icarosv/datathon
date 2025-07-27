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
    # Filtrar registros dessa vaga
    df = df_features[df_features["id_vaga"] == vaga_id].copy()
    if df.empty:
        return df

    # Guardar ids para retorno final
    ids = df[["id_vaga", "id_applicant"]].copy()

    # Remove coluna 'classificacao' se existir
    if "classificacao" in df.columns:
        df = df.drop(columns=["classificacao"])

    # Conversão de colunas não numéricas
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": 1, "false": 0, "sim": 1, "não": 0})
                ).fillna(0)

    # Substitui NaN por 0
    df = df.fillna(0)

    # Conjunto de features
    X = df.drop(["id_vaga", "id_applicant"], axis=1)

    # Calcular probabilidades
    proba_df = predict_proba(X)

    score_col = "prob_1"
    if score_col not in proba_df.columns:
        model = load_model()
        raise ValueError(
            f"Classe positiva '{positive_class}' não encontrada no modelo. "
            f"Classes disponíveis: {list(model.classes_)}"
        )

    ids["score"] = proba_df[score_col].values
    return ids.sort_values("score", ascending=False).head(top_n)
