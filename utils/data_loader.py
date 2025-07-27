import pandas as pd
import streamlit as st
import gdown
from pathlib import Path

# ─── Constantes de diretórios ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
DATA_ROOT     = BASE_DIR / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
FLAT_DIR      = DATA_ROOT / "flat"
FETCH_FLAG    = Path.home() / ".cache" / "datathon" / ".data_fetched"

# ─── Função de download dos dados ───────────────────────────────────────────────
def _fetch_data_folder():
    # Garante existência das pastas
    for d in (DATA_ROOT, FLAT_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if FETCH_FLAG.exists():
        return

    try:
        folder_id = st.secrets["drive"]["data_folder_id"]
    except KeyError:
        st.warning("⚠️ `drive.data_folder_id` faltando em secrets.toml")
        return

    try:
        # Baixa toda a estrutura dentro de data/
        gdown.download_folder(
            id=folder_id,
            output=str(DATA_ROOT),
            quiet=False,
            use_cookies=False,
        )
        # Marca que o download já ocorreu
        FETCH_FLAG.parent.mkdir(parents=True, exist_ok=True)
        FETCH_FLAG.write_text("ok")
    except Exception as e:
        st.error(f"Erro ao baixar dados do Drive: {e}")

# Dispara o download na importação do módulo
_fetch_data_folder()

# ─── Helper seguro para leitura de CSV ─────────────────────────────────────────
def _safe_read_csv(path: Path, sep: str = ";") -> pd.DataFrame:
    if not path.exists():
        st.error(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=sep)
    except pd.errors.ParserError:
        return pd.read_csv(path)

# ─── Funções de carregamento com cache ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_vagas(csv_filename: str = "vagas.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / csv_filename
    return _safe_read_csv(path)

@st.cache_data(show_spinner=False)
def load_applicants(csv_filename: str = "applicants.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / csv_filename
    return _safe_read_csv(path)

@st.cache_data(show_spinner=False)
def load_prospects(csv_filename: str = "prospects.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / csv_filename
    return _safe_read_csv(path)

@st.cache_data(show_spinner=False)
def load_flat(csv_filename: str) -> pd.DataFrame:
    path = FLAT_DIR / csv_filename
    return _safe_read_csv(path)

@st.cache_data(show_spinner=False)
def load_processed_dataset(filename: str = "dataset_for_model.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    df = _safe_read_csv(path)
    if not df.empty:
        # Ajusta coluna de índice
        df.set_index(df.columns[0], inplace=True)
    return df

# ─── Funções auxiliares para busca por ID ───────────────────────────────────────
def get_record_by_id(df: pd.DataFrame, record_id: int, id_col: str = "id") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[df[id_col] == record_id]


def get_vaga_by_id(vaga_id: int, id_col: str = "id") -> pd.DataFrame:
    return get_record_by_id(load_vagas(), vaga_id, id_col)


def get_applicant_by_id(applicant_id: int, id_col: str = "id") -> pd.DataFrame:
    return get_record_by_id(load_applicants(), applicant_id, id_col)


def get_flat_by_id(csv_filename: str, record_id: int, id_col: str = "id") -> pd.DataFrame:
    return get_record_by_id(load_flat(csv_filename), record_id, id_col)
