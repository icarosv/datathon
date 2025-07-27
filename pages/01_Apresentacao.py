import streamlit as st
from pathlib import Path
import base64

# Caminho do logo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
logo_path = PROJECT_ROOT / "assets" / "logo_dashboard.png"

# Se o logo existir, exibe logo + título na mesma linha
if logo_path.exists():
    with open(logo_path, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
            <img src="data:image/png;base64,{encoded}" 
                 style="width:60px; height:60px; border-radius:50%; object-fit:cover;" />
            <h1 style="margin: 0;">Dashboard de Avaliação de Candidatos</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("🎯 Dashboard de Avaliação de Candidatos")

# Descrição geral
st.markdown(
    """
    Bem-vindo ao **Dashboard de Avaliação de Candidatos**! Aqui você poderá:

    - **Pesquisar Vagas**: encontre detalhes de vagas pelo seu `ID`.
    - **Pesquisar Aplicantes**: visualize o perfil de candidatos usando o `ID`.
    - **Predição**: obtenha o ranking dos melhores candidatos para uma vaga específica, baseado em nosso modelo de Machine Learning.
    - **Chat Entrevista**: receba sugestões de perguntas para conduzir entrevistas.

    Use o menu lateral para navegar entre as funcionalidades.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Como funciona a predição")
    st.markdown(
        """
        1. Acesse a página **Predição**.
        2. Informe o **ID da vaga**.
        3. Nosso modelo avalia os perfis dos aplicantes.
        4. Serão apresentadas as 10 melhores recomendações com **scores de adequação**.
        """
    )

with col2:
    st.subheader("Chat para entrevistas")
    st.markdown(
        """
        - Na página **Chat Entrevista**, digite o **ID do(a) aplicante** e o **ID da vaga**.
        - Você receberá sugestões de **perguntas personalizadas** para conduzir a entrevista.
        """
    )

st.markdown("---")
st.caption("Versão do Dashboard: v1.0.0")
