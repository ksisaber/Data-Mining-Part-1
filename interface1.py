import streamlit as st
from part1 import central_tendency, quantiles, missing_unique
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path

# Fonction de chargement des données avec cache
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # Optimisation des types de données pour gagner de la mémoire
    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = data[col].astype("float32")
    for col in data.select_dtypes(include=["int64"]).columns:
        data[col] = data[col].astype("int32")
    return data

def main():
    # Titre principal
    st.title("Projet Data Mining")
    st.sidebar.title("Navigation")

    # Initialisation de l'historique des données
    if "data_history" not in st.session_state:
        st.session_state["data_history"] = []

    # === Partie 1 : Importation et manipulation des données ===
    st.header("1. Importation et Manipulation des Données")

    # Importer un fichier CSV
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    # Charger les données une seule fois dans `st.session_state`
    if uploaded_file:
        if not st.session_state["data_history"]:
            data = load_data(uploaded_file)
            st.session_state["data_history"].append(data)
            st.success("Données chargées avec succès.")

    # Vérifier si des données sont disponibles dans `session_state`
    if st.session_state["data_history"]:

        data = st.session_state["data_history"][-1]

        st.subheader("Aperçu des Données")
        st.dataframe(data.head(100))

        # Ajouter un compteur pour forcer le rafraîchissement
        if "rerun_counter" not in st.session_state:
            st.session_state["rerun_counter"] = 0

        # Bouton pour annuler la dernière opération
        if len(st.session_state["data_history"]) >= 1:
            if st.button("Annuler la dernière opération"):
                st.session_state["data_history"].pop()
                st.session_state["rerun_counter"] += 1
                st.experimental_set_query_params(rerun=st.session_state["rerun_counter"])


        # Bouton pour sauvegarder les données modifiées
        if st.button("Télécharger les données actuelles"):
            st.download_button(
                label="Télécharger CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name="dataset_modifié.csv",
                mime="text/csv",
            )

        # Modification/Suppression d'instances
        st.subheader("Modifier ou Supprimer des Instances")
        row_idx = st.number_input("Indice de la ligne à modifier/supprimer", min_value=0, max_value=len(data) - 1, step=1)
        action = st.selectbox("Action", ["Modifier", "Supprimer"])

        if action == "Modifier":
            col_name = st.selectbox("Choisir une colonne à modifier", data.columns)
            new_value = st.text_input("Nouvelle valeur")
            if st.button("Appliquer la modification"):
                data.at[row_idx, col_name] = new_value
                st.session_state["data_history"].append(data.copy())
                st.success("Modification appliquée.")
                st.dataframe(data.head(100))
        elif action == "Supprimer":
            if st.button("Supprimer la ligne"):
                data = data.drop(index=row_idx).reset_index(drop=True)
                st.session_state["data_history"].append(data.copy())
                st.success("Ligne supprimée.")
                st.dataframe(data.head(100))

    # === Partie 2 : Description globale ===
    if st.session_state["data_history"]:
        st.header("2. Description Globale du Dataset")

        st.write(f"**Dimensions**: {data.shape[0]} lignes, {data.shape[1]} colonnes")
        st.subheader("Statistiques Descriptives")
        st.write(data.describe())
        st.subheader("Valeurs Manquantes")
        st.write(data.isnull().sum())
        st.subheader("Valeurs Uniques")
        st.write(data.nunique())

    # === Partie 3 : Analyse des Attributs ===
    if st.session_state["data_history"]:
        st.header("3. Analyse des Attributs")

        # Sélectionner une colonne pour l'analyse
        selected_col = st.selectbox("Choisir une colonne numérique pour l'analyse", data.select_dtypes(include=[float, int]).columns)

        # Infos générales
        if st.checkbox("Afficher les infos générales"):
            st.markdown("### **Infos Générales sur la Colonne**")
            col1, col2 = st.columns(2)

            # Calcul des statistiques générales
            std_dev = data[selected_col].std()  # Écart-type
            variance = data[selected_col].var()  # Variance
            missing_values = data[selected_col].isnull().sum()  # Valeurs manquantes
            unique_values = data[selected_col].nunique()  # Valeurs uniques

            # Affichage dans deux colonnes
            with col1:
                st.metric(label="Écart-type", value=f"{std_dev:.2f}")
                st.metric(label="Variance", value=f"{variance:.2f}")
            with col2:
                st.metric(label="Valeurs manquantes", value=f"{missing_values}")
                st.metric(label="Valeurs uniques", value=f"{unique_values}")

            st.markdown("### **Mesures de Tendance Centrale**")
            mean, median, mode, symetric = central_tendency(data, selected_col)

            # Vérification et formatage du mode
            if isinstance(mode, pd.Series) or isinstance(mode, list):
                mode_value = ", ".join([f"{m:.2f}" for m in mode])  # Affiche tous les modes formatés
            else:
                mode_value = f"{mode:.2f}"

            # Utilisation de colonnes pour une meilleure présentation
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Moyenne", value=f"{mean:.2f}")
            with col2:
                st.metric(label="Médiane", value=f"{median:.2f}")
            with col3:
                st.metric(label="Mode", value=mode_value)
            with col4:
                st.metric(label="Symétrie", value="Oui" if symetric else "Non")

        if st.checkbox("Afficher l'Histogramme"):
            fig, ax = plt.subplots()
            sns.histplot(data[selected_col], kde=True, bins=10, color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(f"Histogramme de {selected_col}")
            st.pyplot(fig)

    # === Partie 4 : Analyse entre Attributs ===
    if st.session_state["data_history"]:
        st.header("4. Analyse entre Attributs")

        # Sélectionner deux colonnes pour analyser les corrélations
        st.subheader("Corrélations entre deux attributs")
        col1 = st.selectbox("Choisir la première colonne", data.select_dtypes(include=[float, int]).columns, key="col1")
        col2 = st.selectbox("Choisir la deuxième colonne", data.select_dtypes(include=[float, int]).columns, key="col2")
        if col1 == col2:
            st.error("Erreur : Colonnes identiques, veuillez choisir 2 colonnes différentes")
        else:
            if st.button("Afficher le Scatter Plot"):
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[col1], y=data[col2], ax=ax)
                ax.set_title(f"Corrélation entre {col1} et {col2}")
                st.pyplot(fig)

# Lancer l'application
if __name__ == "__main__":
    main()
