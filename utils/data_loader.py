import pandas as pd
from pathlib import Path
import streamlit as st

# Diretório base onde estão os dados processados
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
FLAT_DIR = BASE_DIR / "data" / "flat"

def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV verificando se o arquivo existe antes."""
    if not path.exists():
        st.error(f"Arquivo de dados não encontrado: {path}")
        return pd.DataFrame()
    return pd.read_csv(path,  sep=';')


@st.cache_data(show_spinner=False)
def load_vagas(csv_filename: str = "vagas.csv") -> pd.DataFrame:
    """Carrega o DataFrame de vagas."""
    return _safe_read_csv(DATA_DIR / csv_filename)


@st.cache_data(show_spinner=False)
def load_applicants(csv_filename: str = "applicants.csv") -> pd.DataFrame:
    """Carrega o DataFrame de aplicantes."""
    return _safe_read_csv(DATA_DIR / csv_filename)


def get_record_by_id(df: pd.DataFrame, record_id: int, id_col: str = "id") -> pd.DataFrame:
    """
    Retorna as linhas do DataFrame onde a coluna id_col corresponde ao record_id.
    """
    if df.empty:
        return pd.DataFrame()
    return df[df[id_col] == record_id]


def get_vaga_by_id(vaga_id: int, id_col: str = "id") -> pd.DataFrame:
    """
    Retorna a vaga com o ID informado.
    """
    return get_record_by_id(load_vagas(), vaga_id, id_col)


def get_applicant_by_id(applicant_id: int, id_col: str = "id") -> pd.DataFrame:
    """
    Retorna o aplicante com o ID informado.
    """
    return get_record_by_id(load_applicants(), applicant_id, id_col)

@st.cache_data(show_spinner=False)
def load_flat_flat(csv_filename: str) -> pd.DataFrame:
    """
    Carrega um CSV 'flat' diretamente da pasta data/raw/.
    Ideal para inspecionar dados originais exportados de JSON/ZIP.
    """
    return _safe_read_csv(FLAT_DIR / csv_filename)


def get_flat_by_id(csv_filename: str, record_id: int, id_col: str = "id") -> pd.DataFrame:
    """
    Busca um registro específico em um CSV raw (não processado).
    """
    df = load_flat_flat(csv_filename)
    if df.empty:
        return df
    return df[df[id_col] == record_id]


@st.cache_data(show_spinner=False)
def load_raw_flat(csv_filename: str) -> pd.DataFrame:
    """
    Carrega um CSV 'flat' diretamente da pasta data/flat.
    """
    path = FLAT_DIR / csv_filename
    if not path.exists():
        st.error(f"Arquivo flat não encontrado: {path}")
        return pd.DataFrame()
    # Detecta separador (a maioria será ;)
    try:
        return pd.read_csv(path, sep=';')
    except Exception:
        return pd.read_csv(path)


def get_raw_by_id(csv_filename: str, record_id: int, id_col: str = "id") -> pd.DataFrame:
    """
    Busca um registro específico em um CSV flat (não processado).
    """
    df = load_raw_flat(csv_filename)
    if df.empty:
        return df
    return df[df[id_col] == record_id]

@st.cache_data
def load_processed_dataset(filename: str = "dataset_for_model.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path, sep=";")