import streamlit as st
import pandas as pd
from part2 import (
    outlier,
    normalize_data,
    discretization,
    eliminate_redundancies,
    aggregate_by_season,
)

# === Fonction de chargement des données avec cache ===
@st.cache_data
def load_data(uploaded_file):
    # Charger les données en mémoire
    data = pd.read_csv(uploaded_file)
    # Optimiser les types de données pour économiser la mémoire
    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = data[col].astype("float32")
    for col in data.select_dtypes(include=["int64"]).columns:
        data[col] = data[col].astype("int32")
    return data

# === Fonction principale ===
def main():
    st.title("Partie 2 : Prétraitement Avancé")

    # Chargement des données
    uploaded_file = st.file_uploader("Importer le fichier", type=["csv"])
    if uploaded_file:
        if "data" not in st.session_state:
            st.session_state["data"] = load_data(uploaded_file)
            st.success("Données chargées avec succès.")

    # Utilisation des données depuis `session_state`
    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write(f"**Dimensions des données :** {data.shape[0]} lignes, {data.shape[1]} colonnes")
        st.dataframe(data.head(100))  # Afficher un aperçu limité

        # Agrégation par saisons
        if st.checkbox("Réduction des données par agrégation saisonnière"):
            if "season_data" not in st.session_state:
                st.session_state["season_data"] = aggregate_by_season(data)
            st.write("Données agrégées par saisons :")
            st.dataframe(st.session_state["season_data"].head(100))

        # Gestion des valeurs aberrantes
        if st.checkbox("Gestion des Outliers et des Valeurs Manquantes"):
            st.markdown("### Gestion des Outliers")
            outlier_method = st.selectbox("Méthode pour traiter les outliers", ["zscore", "IQR", "Clipping", "log"])
            selected_cols = st.multiselect("Colonnes à traiter", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la gestion des outliers"):
                st.session_state["data"] = outlier(data, method=outlier_method, cols=selected_cols)
                st.success("Gestion des outliers appliquée.")
                st.dataframe(st.session_state["data"].head(100))

        # Normalisation
        if st.checkbox("Normalisation des données"):
            st.markdown("### Normalisation")
            norm_method = st.radio("Méthode de normalisation", ["minmax", "zscore"])
            selected_cols = st.multiselect("Colonnes à normaliser", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la normalisation"):
                st.session_state["data"] = normalize_data(data, method=norm_method, cols=selected_cols)
                st.success("Normalisation appliquée.")
                st.dataframe(st.session_state["data"].head(100))

        # Discrétisation
        if st.checkbox("Discrétisation des données"):
            st.markdown("### Discrétisation")
            disc_method = st.radio("Méthode de discrétisation", ["equal_frequency", "equal_width"])
            num_bins = st.slider("Nombre de bins", min_value=2, max_value=10, value=5)
            selected_cols = st.multiselect("Colonnes à discrétiser", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la discrétisation"):
                st.session_state["data"] = discretization(data, cols=selected_cols, num_bins=num_bins, method=disc_method)
                st.success("Discrétisation appliquée.")
                st.dataframe(st.session_state["data"].head(100))

        # Réduction des redondances
        if st.checkbox("Réduction des Redondances"):
            st.markdown("### Réduction des Redondances")
            red_method = st.radio("Méthode", ["horizontal", "vertical"])

            if st.button("Appliquer la réduction"):
                st.session_state["data"] = eliminate_redundancies(data, method=red_method)
                st.success("Réduction des redondances appliquée.")
                st.dataframe(st.session_state["data"].head(100))

        # Téléchargement des données traitées
        if st.button("Télécharger les données traitées"):
            st.download_button(
                label="Télécharger CSV",
                data=st.session_state["data"].to_csv(index=False).encode("utf-8"),
                file_name="data_preprocessed.csv",
                mime="text/csv",
            )

# Lancer l'application
if __name__ == "__main__":
    main()
