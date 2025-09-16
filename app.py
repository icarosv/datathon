import streamlit as st
from pathlib import Path

# Configuração do layout
st.set_page_config(
    page_title="Dashboard de Avaliação de Candidatos",
    page_icon="🎯",
    layout="wide",
)

# Caminho para o logo
PROJECT_ROOT = Path(__file__).resolve().parent
logo_path = PROJECT_ROOT / "assets" / "logo_dashboard.png"

# Barra lateral
st.sidebar.title("Navegação")
st.sidebar.markdown(
    """
    Use o menu **Pages** (no canto superior esquerdo ou na barra lateral)
    para navegar entre:
    - **Apresentação**
    - **Pesquisar Vagas**
    - **Pesquisar Aplicantes**
    - **Predição**
    - **Chat Entrevista**
    """
)

# Mostrar logo e título principal na home (quando app.py é aberto diretamente)
if logo_path.exists():
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(str(logo_path), width=80)
    with col_title:
        st.title("🎯 Dashboard de Avaliação de Candidatos")

st.markdown(
    """
    Este é o aplicativo principal do Dashboard.  
    Use as páginas listadas na barra lateral ou na pasta **pages/** para acessar
    as funcionalidades específicas.
    """
)


