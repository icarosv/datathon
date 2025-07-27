import streamlit as st
from utils.data_loader import load_applicants, get_applicant_by_id

# Página de pesquisa de aplicantes pelo ID
st.title("🔍 Pesquisar Aplicantes")
st.markdown("Digite o **ID** do aplicante para visualizar seus detalhes armazenados.")

# Carrega todos os aplicantes
applicants_df = load_applicants()

if applicants_df.empty:
    st.error("Nenhum aplicante disponível no momento.")
else:
    # Limites de ID para facilitar entrada
    min_id = int(applicants_df['id'].min())
    max_id = int(applicants_df['id'].max())

    applicant_id = st.number_input(
        label="ID do Aplicante",
        min_value=min_id,
        max_value=max_id,
        step=1,
        value=min_id
    )

    # Botão de busca
    if st.button("Buscar Aplicante"):
        resultado = get_applicant_by_id(applicant_id)
        if resultado.empty:
            st.warning(f"Nenhum aplicante encontrado com ID **{applicant_id}**.")
        else:
            st.subheader(f"Detalhes do Aplicante {applicant_id}")
            
            if len(resultado) == 1:
                row = resultado.iloc[0]
                cols = st.columns(2)
                half = len(row) // 2
                for i, (col, val) in enumerate(row.items()):
                    target = 0 if i < half else 1
                    cols[target].markdown(f"**{col}:** {str(val)}")
            else:
                st.dataframe(resultado, use_container_width=True)

    # Expander com preview
    with st.expander("Ver primeiros 10 aplicantes"):
        st.dataframe(applicants_df.head(10), use_container_width=True)

