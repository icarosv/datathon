import pandas as pd
from pathlib import Path
import streamlit as st
import gdown

BASE_DIR      = Path(__file__).resolve().parents[1]
DATA_ROOT     = BASE_DIR / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
FLAT_DIR      = DATA_ROOT / "flat"
FETCH_FLAG    = Path.home() / ".cache" / "datathon" / ".data_fetched"

def _fetch_data_folder():
    # 1) cria as pastas antes de baixar
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    FLAT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if FETCH_FLAG.exists():
        return

    try:
        folder_id = st.secrets["drive"]["data_folder_id"]
    except KeyError:
        st.warning("⚠️ `drive.data_folder_id` faltando em secrets.toml")
        return

    try:
        # joga tudo dentro de data/
        gdown.download_folder(
            id=folder_id,
            output=str(DATA_ROOT),
            quiet=False,        # false só na primeira vez pra ver o log
            use_cookies=False,
        )

        # opcional: debug visual da árvore de arquivos
        for p in sorted(DATA_ROOT.rglob("*")):
            st.write("📄", p.relative_to(BASE_DIR))

        # marca que já baixou
        FETCH_FLAG.parent.mkdir(parents=True, exist_ok=True)
        FETCH_FLAG.write_text("ok")
    except Exception as e:
        st.error(f"Erro ao baixar dados do Drive: {e}")

@st.cache_data(show_spinner=False)
def load_vagas() -> pd.DataFrame:
    _fetch_data_folder()
    return _safe_read_csv(PROCESSED_DIR / "vagas.csv")

@st.cache_data(show_spinner=False)
def load_flat(csv_filename: str) -> pd.DataFrame:
    path = FLAT_DIR / csv_filename
    if not path.exists():
        st.error(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=";")
    except Exception as e:
        st.error(f"Erro ao ler {csv_filename}: {e}")
        return pd.DataFrame()

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
