import streamlit as st

relatorios = ['A', 'B', "C"]

relatorio = st.select_box(relatorios)

load = st.button("Carregar")

if load:

    # Carregar relatorio
    
    if relatorio == 'A':
        st.write("Relatório A")
    elif relatorio == 'B':
        st.write("Relatório B")
    else:
        st.write("Relatório C")