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
        if "data_history" not in st.session_state:
            st.session_state["data_history"] = []  # Historique des versions des données
            data = load_data(uploaded_file)
            st.session_state["data_history"].append(data)  # Ajouter la version initiale
            st.success("Données chargées avec succès.")

    # Vérification des données dans l'historique
    if "data_history" in st.session_state and st.session_state["data_history"]:
        data = st.session_state["data_history"][-1]  # Dernière version des données
        st.write(f"**Dimensions des données :** {data.shape[0]} lignes, {data.shape[1]} colonnes")
        st.dataframe(data.head(500))  # Afficher un aperçu limité

        # === Annuler la dernière opération ===
        if st.button("Annuler la dernière opération"):
            if len(st.session_state["data_history"]) > 1:
                st.session_state["data_history"].pop()  # Supprime la dernière version
                st.success("Dernière opération annulée.")
            else:
                st.warning("Aucune opération à annuler.")

        # === Agrégation par saisons ===
        if st.checkbox("Réduction des données par agrégation saisonnière"):
            if st.button("Appliquer l'agrégation par saisons"):
                aggregated_data = aggregate_by_season(data)
                st.session_state["data_history"].append(aggregated_data)  # Ajouter la nouvelle version
                st.success("Agrégation par saisons appliquée.")
                st.write(f"**Dimensions après agrégation :** {aggregated_data.shape[0]} lignes, {aggregated_data.shape[1]} colonnes")
                st.dataframe(aggregated_data.head(500))

        # === Gestion des valeurs aberrantes ===
        if st.checkbox("Gestion des Outliers et des Valeurs Manquantes"):
            st.markdown("### Gestion des Outliers")
            outlier_method = st.selectbox("Méthode pour traiter les outliers", ["zscore", "IQR", "Clipping", "log"])
            selected_cols = st.multiselect("Colonnes à traiter", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la gestion des outliers"):
                outlier_data = outlier(data, method=outlier_method, cols=selected_cols)
                st.session_state["data_history"].append(outlier_data)  # Ajouter la nouvelle version
                st.success("Gestion des outliers appliquée.")
                st.dataframe(outlier_data.head(500))

        # === Normalisation ===
        if st.checkbox("Normalisation des données"):
            st.markdown("### Normalisation")
            norm_method = st.radio("Méthode de normalisation", ["minmax", "zscore"])
            selected_cols = st.multiselect("Colonnes à normaliser", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la normalisation"):
                normalized_data = normalize_data(data, method=norm_method, cols=selected_cols)
                st.session_state["data_history"].append(normalized_data)  # Ajouter la nouvelle version
                st.success("Normalisation appliquée.")
                st.dataframe(normalized_data.head(500))

        # === Discrétisation ===
        # === Discrétisation ===
        if st.checkbox("Discrétisation des données"):
            st.markdown("### Discrétisation")
            disc_method = st.radio("Méthode de discrétisation", ["equal_frequency", "equal_width"])
            num_bins = st.slider("Nombre de bins", min_value=2, max_value=10, value=5)
            selected_cols = st.multiselect("Colonnes à discrétiser", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la discrétisation"):
                discretized_data = discretization(data, cols=selected_cols, num_bins=num_bins, method=disc_method)
                st.session_state["data_history"].append(discretized_data)  # Ajouter la nouvelle version
                st.success("Discrétisation appliquée.")
                st.dataframe(discretized_data[[f"{col}_bins" for col in selected_cols]].head(500))


        # === Réduction des redondances ===
        if st.checkbox("Réduction des Redondances"):
            st.markdown("### Réduction des Redondances")
            red_method = st.radio("Méthode", ["horizontal", "vertical"])

            if st.button("Appliquer la réduction des redondances"):
                reduced_data = eliminate_redundancies(data, method=red_method)
                st.session_state["data_history"].append(reduced_data)
                st.success("Réduction des redondances appliquée.")
                st.dataframe(reduced_data.head(100))

        # Téléchargement des données traitées
        if st.button("Télécharger les données traitées"):
            st.download_button(
                label="Télécharger CSV",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="data_preprocessed.csv",
                mime="text/csv",
            )

# Lancer l'application
if __name__ == "__main__":
    main()
