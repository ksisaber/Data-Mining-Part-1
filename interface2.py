from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from part2 import (
    outlier,
    normalize_data,
    discretization,
    eliminate_redundancies,
    aggregate_by_season,
)

# Chargement de données avec cache
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # 64 vers 32
    for col in data.select_dtypes(include=["float64"]).columns:
        data[col] = data[col].astype("float32")
    for col in data.select_dtypes(include=["int64"]).columns:
        data[col] = data[col].astype("int32")
    return data

# === Fonction principale ===
def main():
    st.title("Partie 2 : Prétraitement Avancé")

    uploaded_file = st.file_uploader("Importer le fichier", type=["csv"])
    if uploaded_file:
        if "data_history" not in st.session_state:
            st.session_state["data_history"] = [] 
            data = load_data(uploaded_file)
            st.session_state["data_history"].append(data)
            st.success("Données chargées avec succès.")

    # Vérification des données dans l'historique
    if "data_history" in st.session_state and st.session_state["data_history"]:
        data = st.session_state["data_history"][-1]
        st.write(f"**Dimensions des données :** {data.shape[0]} lignes, {data.shape[1]} colonnes")
        st.dataframe(data.head(500))

        # Annuler la dernière opération
        if st.button("Annuler la dernière opération"):
            if len(st.session_state["data_history"]) > 1:
                st.session_state["data_history"].pop()
                st.success("Dernière opération annulée.")
            else:
                st.warning("Aucune opération à annuler.")

        # Agrégation par saisons
        st.header("Réduction des données par aggrégation saisonnière")
        if st.checkbox("Réduction des données par agrégation saisonnière"):
            if st.button("Appliquer l'agrégation par saisons"):
                aggregated_data = aggregate_by_season(data)
                st.session_state["data_history"].append(aggregated_data)
                st.success("Agrégation par saisons appliquée.")
                st.write(f"**Dimensions après agrégation :** {aggregated_data.shape[0]} lignes, {aggregated_data.shape[1]} colonnes")
                st.dataframe(aggregated_data.head(500))

        # Gestion des valeurs aberrantes
        st.header("Gestion des Outliers")
        if st.checkbox("Gestion des Outliers"):
            st.markdown("### Gestion des Outliers")
            outlier_method = st.selectbox("Méthode pour traiter les outliers", ["zscore", "IQR", "Clipping", "log"])
            selected_cols = st.multiselect("Colonnes à traiter", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la gestion des outliers"):
                outlier_data = outlier(data, method=outlier_method, cols=selected_cols)
                st.session_state["data_history"].append(outlier_data)
                st.success("Gestion des outliers appliquée.")
                st.dataframe(outlier_data.head(500))

        # Gestion des valeurs manquantes
        st.header("Gestion des valeurs manquantes")
        if st.checkbox("Gestion des Valeurs Manquantes"):
            st.markdown("### Gestion des Valeurs Manquantes")
            missing_option = st.selectbox(
                "Méthode de gestion des valeurs manquantes",
                [
                    "Remplir avec une constante",
                    "Remplir avec la moyenne",
                    "Remplir avec la médiane",
                    "Remplir avec le mode",
                    "Supprimer les lignes avec des valeurs manquantes",
                    "Supprimer les colonnes avec des valeurs manquantes",
                ],
            )
            selected_cols = st.multiselect("Colonnes à traiter", data.columns)

            updated_data = data.copy()

            # Calcul des valeurs manquantes
            initial_missing = updated_data[selected_cols].isnull().sum().sum()

            if missing_option == "Remplir avec une constante":
                constant_value = st.text_input("Valeur constante pour remplacement")
                if st.button("Appliquer"):
                    if constant_value:
                        updated_data[selected_cols] = updated_data[selected_cols].fillna(constant_value)
                        st.session_state["data_history"].append(updated_data)
                        treated_count = initial_missing - updated_data[selected_cols].isnull().sum().sum()
                        st.success(f"Valeurs manquantes remplacées par {constant_value}. Total traité : {treated_count}.")
            elif missing_option == "Remplir avec la moyenne":
                if st.button("Appliquer"):
                    updated_data[selected_cols] = updated_data[selected_cols].fillna(updated_data[selected_cols].mean())
                    st.session_state["data_history"].append(updated_data)
                    treated_count = initial_missing - updated_data[selected_cols].isnull().sum().sum()
                    st.success(f"Valeurs manquantes remplacées par la moyenne. Total traité : {treated_count}.")
            elif missing_option == "Remplir avec la médiane":
                if st.button("Appliquer"):
                    updated_data[selected_cols] = updated_data[selected_cols].fillna(updated_data[selected_cols].median())
                    st.session_state["data_history"].append(updated_data)
                    treated_count = initial_missing - updated_data[selected_cols].isnull().sum().sum()
                    st.success(f"Valeurs manquantes remplacées par la médiane. Total traité : {treated_count}.")
            elif missing_option == "Remplir avec le mode":
                if st.button("Appliquer"):
                    for col in selected_cols:
                        mode_value = updated_data[col].mode()[0]
                        updated_data[col] = updated_data[col].fillna(mode_value)
                    st.session_state["data_history"].append(updated_data)
                    treated_count = initial_missing - updated_data[selected_cols].isnull().sum().sum()
                    st.success(f"Valeurs manquantes remplacées par le mode. Total traité : {treated_count}.")
            elif missing_option == "Supprimer les lignes avec des valeurs manquantes":
                if st.button("Appliquer"):
                    initial_rows = updated_data.shape[0]
                    updated_data = updated_data.dropna()
                    st.session_state["data_history"].append(updated_data)
                    removed_rows = initial_rows - updated_data.shape[0]
                    st.success(f"Lignes supprimées. Total de lignes supprimées : {removed_rows}.")
            elif missing_option == "Supprimer les colonnes avec des valeurs manquantes":
                if st.button("Appliquer"):
                    initial_cols = updated_data.shape[1]
                    updated_data = updated_data.dropna(axis=1)
                    st.session_state["data_history"].append(updated_data)
                    removed_cols = initial_cols - updated_data.shape[1]
                    st.success(f"Colonnes supprimées. Total de colonnes supprimées : {removed_cols}.")

            # Affichage du dataframe mis à jour
            st.dataframe(updated_data.head(500))

        # Normalisation
        st.header("Normalisation des données")
        if st.checkbox("Normalisation des données"):
            st.markdown("### Normalisation")
            norm_method = st.radio("Méthode de normalisation", ["minmax", "zscore"])
            selected_cols = st.multiselect("Colonnes à normaliser", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la normalisation"):
                normalized_data = normalize_data(data, method=norm_method, cols=selected_cols)
                st.session_state["data_history"].append(normalized_data)
                st.success("Normalisation appliquée.")
                st.dataframe(normalized_data.head(500))

        # Discrétisation
        st.header("Discrétisation des données")
        if st.checkbox("Discrétisation des données"):
            st.markdown("### Discrétisation")
            disc_method = st.radio("Méthode de discrétisation", ["equal_frequency", "equal_width"])
            num_bins = st.slider("Nombre de bins", min_value=2, max_value=10, value=5)
            selected_cols = st.multiselect("Colonnes à discrétiser", data.select_dtypes(include=[float, int]).columns)

            if st.button("Appliquer la discrétisation"):
                discretized_data = discretization(data, cols=selected_cols, num_bins=num_bins, method=disc_method)
                st.session_state["data_history"].append(discretized_data)
                st.success("Discrétisation appliquée.")
                st.dataframe(discretized_data.head(500))

        # Réduction des redondances
        st.header("Réduction des redondances")
        if st.checkbox("Réduction des Redondances"):
            st.markdown("### Réduction des Redondances")
            red_method = st.radio("Méthode", ["horizontal", "vertical"])

            if st.button("Appliquer la réduction des redondances"):
                reduced_data = eliminate_redundancies(data, method=red_method)
                st.session_state["data_history"].append(reduced_data)
                st.success("Réduction des redondances appliquée.")
                st.dataframe(reduced_data.head(100))

        # Téléchargement des données traitées
        st.header("Télécharger les données traitées")
        if st.button("Télécharger les données traitées"):
            st.download_button(
                label="Télécharger CSV",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="data_preprocessed.csv",
                mime="text/csv",
            )
        

        # Vérification si des données sont chargées
        if st.session_state["data_history"] and not st.session_state["data_history"][-1].empty:
            # Checkbox pour afficher la carte
            if st.checkbox("Afficher Carte"):
                st.header("Carte d'Intensité Basée sur les Propriétés")

                # Récupération des données actuelles
                map_df = st.session_state["data_history"][-1]

                # Choix du type de propriété
                prop_type = st.radio("Sélectionner le type de propriété :", ["Propriétés du Sol", "Propriétés Climatiques"])

                # Dropdown pour la sélection de propriété selon le type
                if prop_type == "Propriétés du Sol":
                    prop = st.selectbox("Choisir une propriété du sol :", [
                        "sand % topsoil", "sand % subsoil", "silt % topsoil", "silt % subsoil", "clay % topsoil", 
                        "clay % subsoil", "pH water topsoil", "pH water subsoil", "OC % topsoil", "OC % subsoil", 
                        "N % topsoil", "N % subsoil", "BS % topsoil", "BS % subsoil", "CEC topsoil", "CEC subsoil", 
                        "CEC clay topsoil", "CEC Clay subsoil", "CaCO3 % topsoil", "CaCO3 % subsoil", "BD topsoil", 
                        "BD subsoil", "C/N topsoil", "C/N subsoil"
                    ])
                else:
                    prop = st.selectbox("Choisir une propriété climatique :", ["Rainf", "Tair", "Wind", "Qair", "Snowf", "PSurf"])

                # Sélection de la saison
                season_prop = st.selectbox("Choisir une saison :", ["Spring", "Summer", "Autumn", "Winter"])

                # Sélection de la palette de couleurs
                color_palette = st.selectbox(
                    "Choisir une palette de couleurs :",
                    ["Reds", "Blues", "Greens"]
                )

                # Extraction des latitudes, longitudes et intensité en fonction de la propriété sélectionnée
                try:
                    lats = map_df["lat"].values  # Coordonnées de latitude
                    longs = map_df["lon"].values  # Coordonnées de longitude
                    intensity = map_df[f"{prop}_{season_prop}" if prop_type == "Propriétés Climatiques" else prop].values

                    # Création de la carte avec Cartopy
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax = plt.axes(projection=ccrs.PlateCarree())
                    ax.set_extent([-10, 12, 18, 38])  # Délimitation pour l'Algérie

                    # Ajout des éléments de la carte
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=':')
                    ax.add_feature(cfeature.LAKES, alpha=0.4)

                    # Normalisation des valeurs pour la palette
                    norm = Normalize(vmin=min(intensity), vmax=max(intensity))
                    cmap = plt.get_cmap(color_palette)

                    # Tracer les points d'intensité
                    scatter = ax.scatter(
                        longs, lats, c=intensity, cmap=cmap, norm=norm, s=30, alpha=0.8, edgecolor='none'
                    )

                    # Barre de couleur pour représenter l'intensité
                    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical", label="Intensité")

                    # Titre de la carte
                    plt.title(f"Carte d'Intensité de {prop} en Algérie ({season_prop})")

                    # Afficher la carte dans Streamlit
                    st.pyplot(fig)

                except KeyError:
                    st.error("Les colonnes nécessaires (lat, lon, ou propriétés) ne sont pas présentes dans le dataset.")



# Lancer l'application
if __name__ == "__main__":
    main()
