import streamlit as st
from utils.data_loader import load_raw_flat, get_raw_by_id

# Página de pesquisa de vagas pelo ID
st.title("🔍 Pesquisar Vagas (dados flat)")
st.markdown(
    """
    Esses dados vêm diretamente dos arquivos **flat** (dados brutos extraídos do JSON, sem pré-processamento).
    Digite o **ID** da vaga para visualizar seus detalhes originais.
    """
)

# Nome do CSV flat que você salvou
csv_flat = "vagas_flat.csv"

# Carregar dataset de vagas (flat)
vagas_df = load_raw_flat(csv_flat)

if vagas_df.empty:
    st.error("Nenhuma vaga disponível no momento.")
else:
    # Garantir que existe coluna id
    if "id" not in vagas_df.columns:
        st.error(f"O arquivo {csv_flat} não contém a coluna 'id'.")
    else:
        # Entrada do ID da vaga
        min_id = int(vagas_df['id'].min())
        max_id = int(vagas_df['id'].max())

        vaga_id = st.number_input(
            label="ID da Vaga",
            min_value=min_id,
            max_value=max_id,
            step=1,
            value=min_id,
        )

        # Botão para buscar vaga
        if st.button("Buscar Vaga"):
            resultado = get_raw_by_id(csv_flat, vaga_id)
            if resultado.empty:
                st.warning(f"Nenhuma vaga encontrada com ID **{vaga_id}**.")
            else:
                if len(resultado) == 1:
                    row = resultado.iloc[0]
                    cols = st.columns(2)
                    half = len(row) // 2
                    for i, (col, val) in enumerate(row.items()):
                        target = 0 if i < half else 1
                        cols[target].markdown(f"**{col}:** {val}")
                else:
                    st.dataframe(resultado, use_container_width=True)


        # Mostrar lista de referência
        with st.expander("Ver primeiras 10 vagas (flat)"):
            st.dataframe(vagas_df.head(10), use_container_width=True)

