import streamlit as st
from part1 import central_tendency, quantiles, missing_unique
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path

# Utilisation de cache pour charger les données plus rapidement
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    # 64 vers 32 pour charger notre data plus rapidement
    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = data[col].astype("float32")
    for col in data.select_dtypes(include=["int64"]).columns:
        data[col] = data[col].astype("int32")
    return data

def main():
    st.title("Projet Data Mining")
    # Historique pour annuler les opérations
    if "data_history" not in st.session_state:
        st.session_state["data_history"] = []



    # === Partie 1 : Importation et manipulation des données ===
    st.header("1. Importation et Manipulation des Données")

    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    # Utilisation de st.session state pour charger les données 1 seule fois
    if uploaded_file:
        if not st.session_state["data_history"]:
            data = load_data(uploaded_file)
            st.session_state["data_history"].append(data)
            st.success("Données chargées avec succès.")

    # Vérifier si des données sont disponibles dans session_state
    if st.session_state["data_history"]:

        data = st.session_state["data_history"][-1]

        st.subheader("Aperçu des Données")
        st.dataframe(data.head(100))

        # Forcer le rafraichissement
        if "rerun_counter" not in st.session_state:
            st.session_state["rerun_counter"] = 0

        # Annuler la dernière opération
        if len(st.session_state["data_history"]) >= 1:
            if st.button("Annuler la dernière opération"):
                st.session_state["data_history"].pop()
                st.session_state["rerun_counter"] += 1
                st.experimental_set_query_params(rerun=st.session_state["rerun_counter"])


        # Sauvegarder les données modifiées
        if st.button("Télécharger les données actuelles"):
            st.download_button(
                label="Télécharger CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name="dataset_modifié.csv",
                mime="text/csv",
            )

        # Modification/Suppression d'instances
        st.subheader("Modifier ou Supprimer des Instances")
        if st.checkbox("Modifier/Supprimer des instances"):
            row_idx = st.number_input("Indice de la ligne à modifier/supprimer", min_value=0, max_value=len(data) - 1, step=1)
            choix = st.selectbox("Choisir le traitement : Ligne ou Colonne", ["Ligne", "Colonne"])

            if choix == "Ligne":
                action = st.selectbox("Action", ["Modifier", "Supprimer"], key="action_ligne")

                if action == "Modifier":
                    col_name = st.selectbox("Choisir une colonne à modifier", data.columns, key="col_name_modifier")
                    new_value = st.text_input("Nouvelle valeur")
                    if st.button("Appliquer la modification"):
                        data.at[row_idx, col_name] = new_value
                        st.session_state["data_history"].append(data.copy())
                        st.success(f"Valeur modifiée dans la colonne '{col_name}' à l'indice {row_idx}.")
                        st.dataframe(data.head(100))
                elif action == "Supprimer":
                    if st.button("Supprimer la ligne"):
                        data = data.drop(index=row_idx).reset_index(drop=True)
                        st.session_state["data_history"].append(data.copy())
                        st.success(f"Ligne {row_idx} supprimée.")
                        st.dataframe(data.head(100))

            elif choix == "Colonne":
                action = st.selectbox("Action", ["Modifier", "Supprimer"], key="action_colonne")

                if action == "Modifier":
                    col_name = st.selectbox("Choisir une colonne à modifier", data.columns, key="col_name_modifier_colonne")
                    new_col_name = st.text_input("Nouveau nom pour la colonne", key="new_col_name")
                    if st.button("Appliquer le nouveau nom"):
                        data = data.rename(columns={col_name: new_col_name})
                        st.session_state["data_history"].append(data.copy())
                        st.success(f"La colonne '{col_name}' a été renommée en '{new_col_name}'.")
                        st.dataframe(data.head(100))

                elif action == "Supprimer":
                    col_name = st.selectbox("Choisir une colonne à supprimer", data.columns, key="col_name_supprimer")
                    if st.button("Supprimer la colonne"):
                        data = data.drop(columns=[col_name]).reset_index(drop=True)
                        st.session_state["data_history"].append(data.copy())
                        st.success(f"Colonne '{col_name}' supprimée.")
                        st.dataframe(data.head(100))



    # === Partie 2 : Description globale ===
    if st.session_state["data_history"]:
        st.header("2. Description Globale du Dataset")
        if st.checkbox("Afficher la description globale du dataset"):
            st.write(f"**Dimensions**: {data.shape[0]} lignes, {data.shape[1]} colonnes")
            st.subheader("Statistiques Descriptives")
            st.write(data.describe())
            
            st.subheader("Valeurs Manquantes")
            missing_values = data.isnull().sum()
            st.write("Nombre de valeurs manquantes par colonne :")
            st.dataframe(missing_values[missing_values >= 0])
            
            st.subheader("Valeurs Uniques")
            unique_values = data.nunique()
            st.write("Nombre de valeurs uniques par colonne :")
            st.dataframe(unique_values)



    # === Partie 3 : Analyse des Attributs ===
    if st.session_state["data_history"]:
        st.header("3. Analyse des Attributs")
        
        selected_col = st.selectbox("Choisir une colonne numérique pour l'analyse", data.select_dtypes(include=[float, int]).columns)

        # Infos générales
        if st.checkbox("Afficher les infos générales"):
            st.markdown("### **Infos Générales sur la Colonne**")
            col1, col2 = st.columns(2)

            # Calcul des statistiques générales
            std_dev = data[selected_col].std()
            variance = data[selected_col].var()
            missing_values = data[selected_col].isnull().sum()
            unique_values = data[selected_col].nunique()

            # Affichage
            with col1:
                st.metric(label="Écart-type", value=f"{std_dev:.2f}")
                st.metric(label="Variance", value=f"{variance:.2f}")
            with col2:
                st.metric(label="Valeurs manquantes", value=f"{missing_values}")
                st.metric(label="Valeurs uniques", value=f"{unique_values}")

            st.markdown("### **Mesures de Tendance Centrale**")
            mean, median, mode, symetric = central_tendency(data, selected_col)

            if isinstance(mode, pd.Series) or isinstance(mode, list):
                mode_value = ", ".join([f"{m:.2f}" for m in mode])
            else:
                mode_value = f"{mode:.2f}"

            # Affichage avec col
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Moyenne", value=f"{mean:.2f}")
            with col2:
                st.metric(label="Médiane", value=f"{median:.2f}")
            with col3:
                st.metric(label="Mode", value=mode_value)
            with col4:
                st.metric(label="Symétrie", value="Oui" if symetric else "Non")

                
            st.markdown("### **Mesures de Dispersion et Outliers**")

            # Calcul des quantiles et des bornes
            q, lower, upper, quantile_att = quantiles(data, selected_col)

            # Vérification et formatage des quantiles si c'est une liste ou une série
            if isinstance(q, (pd.Series, list, np.ndarray)):
                q_formatted = ", ".join([f"{val:.2f}" for val in q])
            else:
                q_formatted = f"{q:.2f}"

            # Création d'un tableau pour les quantiles
            quantile_table = pd.DataFrame(
                {
                    "Quantile": ["Min", "1er Quartile", "Médiane", "3e Quartile", "Max"],
                    "Valeur": [f"{val:.2f}" for val in q]
                }
            )

            # Affichage du tableau des quantiles
            st.markdown("#### **Tableau des Quantiles**")
            st.table(quantile_table)

            # Affichage avec col
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Borne Inférieure", value=f"{lower:.2f}")
            with col2:
                st.metric(label="Borne Supérieure", value=f"{upper:.2f}")

            # Affichage des valeurs aberrantes
            st.markdown("#### **Valeurs Aberrantes**")
            outliers = quantile_att

            if not outliers.empty:
                st.dataframe(outliers)
            else:
                st.success("Aucune valeur aberrante détectée.")

        # Visualisations
        st.subheader("Visualisations")
        if st.checkbox("Afficher le Boxplot"):
            fig, ax = plt.subplots()
            sns.boxplot(y=data[selected_col], ax=ax)
            st.pyplot(fig)
            
        if st.checkbox("Afficher l'Histogramme"):
            fig, ax = plt.subplots()
            sns.histplot(data[selected_col], kde=True, bins=10, color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(f"Histogramme de {selected_col}")
            st.pyplot(fig)



    # === Partie 4 : Analyse entre Attributs ===
    if st.session_state["data_history"]:
        st.header("4. Analyse entre Attributs")

        # Sélectionner 2 colonnes pour l'analyse
        st.subheader("Corrélations entre deux attributs")
        if st.checkbox("Afficher la corrélation entre 2 attributs"):
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
