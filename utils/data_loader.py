import pandas as pd
from pathlib import Path
import streamlit as st
import gdown

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parents[1]

# Onde a pasta inteira de dados será baixada e armazenada
DATA_ROOT = BASE_DIR / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
FLAT_DIR = DATA_ROOT / "flat"

# Flag para não baixar repetidamente
FETCH_FLAG = BASE_DIR / ".data_fetched"

def _fetch_data_folder():
    if FETCH_FLAG.exists():
        return
    try:
        folder_id = st.secrets["drive"]["data_folder_id"]
    except KeyError:
        st.warning("⚠️ `data_folder_id` não configurado em secrets.toml.")
        return
    # baixa a pasta inteira data/ no Drive
    gdown.download_folder(
        id=folder_id,
        output=str(BASE_DIR),
        quiet=True,
        use_cookies=False,
    )
    FETCH_FLAG.write_text("ok")
    #st.write(BASE_DIR)


def _safe_read_csv(path: Path, sep=";") -> pd.DataFrame:
    """Lê um CSV verificando se o arquivo existe antes."""
    if not path.exists():
        st.error(f"Arquivo de dados não encontrado: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep)


@st.cache_data(show_spinner=False)
def load_vagas(csv_filename: str = "vagas.csv") -> pd.DataFrame:
    """Carrega o DataFrame de vagas, baixando do Drive se necessário."""
    _fetch_data_folder()
    return _safe_read_csv(PROCESSED_DIR / csv_filename)


@st.cache_data(show_spinner=False)
def load_applicants(csv_filename: str = "applicants.csv") -> pd.DataFrame:
    """Carrega o DataFrame de aplicantes, baixando do Drive se necessário."""
    _fetch_data_folder()
    return _safe_read_csv(PROCESSED_DIR / csv_filename)


@st.cache_data(show_spinner=False)
def load_prospects(csv_filename: str = "prospects.csv") -> pd.DataFrame:
    """Carrega o DataFrame de prospects processados."""
    _fetch_data_folder()
    return _safe_read_csv(PROCESSED_DIR / csv_filename)


def get_record_by_id(df: pd.DataFrame, record_id: int, id_col: str = "id") -> pd.DataFrame:
    """Retorna as linhas do DataFrame onde id_col == record_id."""
    if df.empty:
        return pd.DataFrame()
    return df[df[id_col] == record_id]


def get_vaga_by_id(vaga_id: int, id_col: str = "id") -> pd.DataFrame:
    """Retorna o registro de vaga com ID informado."""
    return get_record_by_id(load_vagas(), vaga_id, id_col)


def get_applicant_by_id(applicant_id: int, id_col: str = "id") -> pd.DataFrame:
    """Retorna o registro de aplicante com ID informado."""
    return get_record_by_id(load_applicants(), applicant_id, id_col)


@st.cache_data(show_spinner=False)
def load_flat(csv_filename: str) -> pd.DataFrame:
    """
    Carrega um CSV 'flat' diretamente da pasta data/flat,
    baixando do Drive se necessário.
    """
    _fetch_data_folder()
    return _safe_read_csv(FLAT_DIR / csv_filename)


def get_flat_by_id(csv_filename: str, record_id: int, id_col: str = "id") -> pd.DataFrame:
    """Busca um registro específico em um CSV flat."""
    df = load_flat(csv_filename)
    if df.empty:
        return df
    return df[df[id_col] == record_id]


@st.cache_data(show_spinner=False)
def load_processed_dataset(filename: str = "dataset_for_model.csv") -> pd.DataFrame:
    """
    Carrega o dataset pronto para o modelo, com índice e colunas corretas.
    """
    # Se você estiver usando fetch do Drive, descomente a próxima linha:
    # _fetch_data_folder()

    path = PROCESSED_DIR / filename
    if not path.exists():
        st.error(f"Arquivo de dataset para modelo não encontrado: {path}")
        return pd.DataFrame()

    # Lê usando o índice salvo (coluna 0) e separador ';'
    return pd.read_csv(path, sep=";", index_col=0)
