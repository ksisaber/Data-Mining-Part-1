import streamlit as st
from streamlit_option_menu import option_menu  # Pour une barre de navigation stylée

st.set_page_config(page_title="Data Mining Project", layout="wide")

# Barre de navigation
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Partie 1 : Analyse des Attributs", "Partie 2 : Prétraitement Avancé"],
        icons=["bar-chart", "gear"],
        menu_icon="menu-app",
        default_index=0,
    )

if selected == "Partie 1 : Analyse des Attributs":
    import interface1 
    interface1.main() 

elif selected == "Partie 2 : Prétraitement Avancé":
    import interface2
    interface2.main()
