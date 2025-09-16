import streamlit as st
from pathlib import Path
import google.generativeai as genai

from utils.data_loader import _fetch_data_folder, get_applicant_by_id, get_vaga_by_id

_fetch_data_folder()
# =====================
# Configuração Gemini
# =====================
api_key = st.secrets.get("gemini", {}).get("api_key")
if not api_key:
    st.error("API key do Gemini não encontrada. Configure em .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

# =====================
# Função para criar o contexto inicial
# =====================
def gerar_contexto_inicial(applicant_id: int, vaga_id: int):
    applicant_df = get_applicant_by_id(applicant_id)
    vaga_df = get_vaga_by_id(vaga_id)

    if applicant_df.empty or vaga_df.empty:
        return None, None, "Não foram encontrados dados para esse candidato ou vaga."

    applicant_info = applicant_df.to_dict(orient="records")[0]
    vaga_info = vaga_df.to_dict(orient="records")[0]

    contexto = f"""
    Você é um especialista em RH.
    Avalie a seguinte vaga e perfil de candidato.
    Use essas informações para responder perguntas e gerar sugestões de perguntas de entrevista.

    **Descrição da vaga:**
    {vaga_info}

    **Perfil do aplicante:**
    {applicant_info}
    """
    return applicant_info, vaga_info, contexto

# =====================
# Inicialização do estado da sessão
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = None

# =====================
# Interface Streamlit
# =====================
st.title("💬 Chat Entrevista (Gemini)")
st.markdown(
    """
    Este chat usa a API do Gemini para ajudar a criar e refinar perguntas de entrevista
    personalizadas para um aplicante e uma vaga específicos.
    """
)

# Inputs para IDs (apenas no início)
if st.session_state.chat is None:
    col1, col2 = st.columns(2)
    with col1:
        applicant_id = st.number_input("ID do Aplicante", min_value=1, step=1)
    with col2:
        vaga_id = st.number_input("ID da Vaga", min_value=1, step=1)

    if st.button("Iniciar Chat"):
        applicant_info, vaga_info, contexto = gerar_contexto_inicial(applicant_id, vaga_id)
        if contexto.startswith("Não foram encontrados"):
            st.error(contexto)
        else:
            # Iniciar chat do Gemini com contexto
            st.session_state.chat = model.start_chat(history=[
                {"role": "user", "parts": [contexto]}
            ])
            st.session_state.messages.append({"role": "user", "content": contexto})
            st.success("Contexto carregado! Agora envie mensagens no chat abaixo.")

# Mostrar histórico
if st.session_state.chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de texto
    if prompt := st.chat_input("Escreva sua pergunta ou pedido..."):
        # Adiciona pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera resposta
        with st.chat_message("assistant"):
            with st.spinner("Gerando resposta..."):
                response = st.session_state.chat.send_message(prompt)
                st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
