import pandas as pd
import os
import gc

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================
NB_LIGNES = 100000
TOP_N_GENRES = 10  # Nombre de genres à conserver (les autres → "Autre")

# ============================================================================
# ÉTAPE 1 : FILTRAGE DES AVIS (>= 10 avis par livre)
# ============================================================================
def extraire_et_filtrer(df):
    print("Filtrage des données (minimum 10 avis par livre)...")
    
    # On cherche la colonne qui identifie le livre (Title ou Id)
    col_item = "Title" if "Title" in df.columns else ("Id" if "Id" in df.columns else None)
    
    if col_item:
        # On compte le nombre d'avis par livre
        counts = df[col_item].value_counts()
        # On garde seulement les livres qui ont au moins 10 avis
        titres_valides = counts[counts >= 10].index
        df_filtree = df[df[col_item].isin(titres_valides)]
        print(f"Nombre d'avis après filtrage (>= 10 avis) : {len(df_filtree)} sur {len(df)} initialement.")
    else:
        print("Avertissement : Colonne Title ou Id non trouvée. Filtrage ignoré.")
        df_filtree = df

    print("\nÉchantillonnage équidistribué...")
    df_temp = pd.DataFrame()
    for i in range(5):
        note = i + 1
        temp = df_filtree.loc[df_filtree["review/score"] == note, :]
        
        nb_dispo = len(temp)
        remplacement = True if nb_dispo < NB_LIGNES else False
        
        echantillon = temp.sample(n=NB_LIGNES, replace=remplacement, random_state=42)
        df_temp = pd.concat([df_temp, echantillon])
        print(f"Note {note} : {NB_LIGNES} avis extraits (Remplacement={'Oui' if remplacement else 'Non'})")
        
    return df_temp

# ============================================================================
# ÉTAPE 2 : ANALYSE DES CATÉGORIES (diagnostic)
# ============================================================================
def analyser_categories(df):
    print("\n--- Analyse des catégories du nouvel échantillon ---")
    
    if "categories" in df.columns:
        categories_top = df["categories"].value_counts().head(10)
        print("Top 10 des catégories les plus représentées :")
        for cat, freq in categories_top.items():
            pourcentage = (freq / len(df)) * 100
            print(f" - {cat} : {freq} avis ({pourcentage:.2f}%)")
    else:
        print("La colonne 'categories' n'est pas présente dans le dataset. Analyse impossible.")
        print("Colonnes disponibles :", df.columns.tolist())

# ============================================================================
# ÉTAPE 3 : FUSION DES GENRES (Left Join + Nettoyage + One-Hot Encoding)
# ============================================================================
def fusionner_genres(df_avis, chemin_books_data="data/books_data.csv"):
    """
    Fusionne les avis avec les métadonnées des livres pour récupérer le genre,
    puis applique un nettoyage strict et un One-Hot Encoding.
    
    Pipeline :
        1. Chargement de books_data.csv (colonnes réduites)
        2. Nettoyage du format "['Genre']" → texte brut
        3. Left Join sur la colonne Title
        4. Gestion des NaN → "Inconnu"
        5. Réduction de la cardinalité (Top N → "Autre")
        6. One-Hot Encoding (dtype int)
    """
    
    # --- 3.1 : Chargement optimisé (seulement 2 colonnes en mémoire) ---
    print("\n--- Fusion des Genres ---")
    print(f"Chargement de '{chemin_books_data}' (colonnes : Title, categories)...")
    df_genres = pd.read_csv(
        chemin_books_data,
        usecols=["Title", "categories"],
        dtype={"Title": str, "categories": str},
        on_bad_lines='skip'
    )
    print(f"  → {len(df_genres)} lignes chargées depuis books_data.csv")
    
    # --- 3.2 : Nettoyage du format brut de la colonne categories ---
    # Le format est typiquement "['Fiction']" ou "['Sci-Fi', 'Romance']"
    # On retire les crochets, guillemets, et on ne garde que le premier genre.
    print("Nettoyage de la colonne 'categories'...")
    
    # Retirer les caractères de liste Python : [ ] '
    df_genres["categories"] = (
        df_genres["categories"]
        .str.replace(r"[\[\]']", "", regex=True)  # Supprime [ ] '
        .str.strip()                               # Supprime les espaces
    )
    
    # Si multi-genres séparés par virgule → ne garder que le premier
    df_genres["categories"] = (
        df_genres["categories"]
        .str.split(",")
        .str[0]
        .str.strip()
    )
    
    # Remplacer les chaînes vides par NaN pour uniformiser le traitement
    df_genres["categories"] = df_genres["categories"].replace("", pd.NA)
    
    # --- 3.3 : Déduplication sur Title (éviter la multiplication de lignes au merge) ---
    nb_avant = len(df_genres)
    df_genres = df_genres.drop_duplicates(subset="Title", keep="first")
    print(f"  → Déduplication : {nb_avant} → {len(df_genres)} livres uniques")
    
    # --- 3.4 : LEFT JOIN (préserve intégralement les 500k lignes d'avis) ---
    nb_avis_avant = len(df_avis)
    print(f"Left Join sur 'Title' ({nb_avis_avant} avis)...")
    df_avis = df_avis.merge(df_genres[["Title", "categories"]], on="Title", how="left")
    
    assert len(df_avis) == nb_avis_avant, \
        f"ERREUR CRITIQUE : Le merge a changé le nombre de lignes ({nb_avis_avant} → {len(df_avis)})"
    print(f"  → ✅ Join terminé. Nombre de lignes préservé : {len(df_avis)}")
    
    # Libérer df_genres de la mémoire (plus besoin)
    del df_genres
    gc.collect()
    
    # --- 3.5 : Gestion des NaN → "Inconnu" ---
    nb_nan = df_avis["categories"].isna().sum()
    df_avis["categories"] = df_avis["categories"].fillna("Inconnu")
    print(f"  → {nb_nan} valeurs manquantes remplacées par 'Inconnu' ({nb_nan/len(df_avis)*100:.1f}%)")
    
    # --- 3.6 : Réduction de la cardinalité (Top N + "Autre") ---
    print(f"\nRéduction de la cardinalité au Top {TOP_N_GENRES} genres...")
    freq_genres = df_avis["categories"].value_counts()
    top_genres = freq_genres.head(TOP_N_GENRES).index.tolist()
    
    # Tous les genres hors Top N → "Autre"
    df_avis["categories"] = df_avis["categories"].where(
        df_avis["categories"].isin(top_genres), other="Autre"
    )
    
    # Affichage du résultat pour vérification
    print(f"\nDistribution finale des genres ({len(df_avis['categories'].unique())} catégories) :")
    for genre, count in df_avis["categories"].value_counts().items():
        pourcentage = (count / len(df_avis)) * 100
        print(f"  • {genre:<25s} : {count:>7d} avis ({pourcentage:5.1f}%)")
    
    # --- 3.7 : One-Hot Encoding (dtype=int pour compatibilité tenseurs) ---
    print("\nOne-Hot Encoding de la colonne 'categories'...")
    genre_dummies = pd.get_dummies(df_avis["categories"], prefix="genre", dtype=int)
    df_avis = pd.concat([df_avis, genre_dummies], axis=1)
    
    # Suppression de la colonne catégorielle source (remplacée par le OHE)
    df_avis.drop(columns=["categories"], inplace=True)
    
    # Vérification : chaque ligne doit avoir exactement un 1 dans les colonnes genre_*
    cols_genre = [c for c in df_avis.columns if c.startswith("genre_")]
    somme_par_ligne = df_avis[cols_genre].sum(axis=1)
    assert (somme_par_ligne == 1).all(), "ERREUR : Certaines lignes n'ont pas exactement un genre actif !"
    print(f"  → ✅ {len(cols_genre)} colonnes genre créées : {cols_genre}")
    print(f"  → ✅ Vérification OHE : chaque ligne a exactement 1 genre actif.")
    
    del genre_dummies
    gc.collect()
    
    return df_avis

# ============================================================================
# ÉTAPE 4 : EXPORT DU CSV FINAL
# ============================================================================
def export_csv(df_a_extraire):
    if not os.path.exists("data"):
        os.mkdir("data")
    
    print("\nMélange des données (Shuffle)...")
    df_a_extraire = df_a_extraire.sample(frac=1, random_state=42).reset_index(drop=True)
    
    nom_fichier = "data/books_rating_500k_filtre.csv"
    df_a_extraire.to_csv(nom_fichier, index=False)
    print(f"\n✅ CSV propre et équidistribué généré avec succès dans '{nom_fichier}'")

# ============================================================================
# MAIN : PIPELINE COMPLET
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE DE PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    # 1. Chargement du dataset brut
    print("\n[1/4] Chargement du dataset initial...")
    df = pd.read_csv("data/Books_rating.csv", sep=",")
    
    # 2. Filtrage (>= 10 avis) + Échantillonnage équidistribué (100k/note)
    print("\n[2/4] Filtrage et échantillonnage...")
    df_resultat = extraire_et_filtrer(df)
    
    # Libérer le DF brut (très volumineux : ~2.8 Go)
    del df
    gc.collect()
    
    # 3. Fusion des genres + One-Hot Encoding
    print("\n[3/4] Fusion des genres et One-Hot Encoding...")
    df_resultat = fusionner_genres(df_resultat)
    
    # 4. Export du CSV enrichi
    print("\n[4/4] Exportation du CSV final...")
    export_csv(df_resultat)
    
    print("\n" + "=" * 60)
    print("  PIPELINE TERMINÉ AVEC SUCCÈS ✅")
    print("=" * 60)