import pandas as pd 
import os

NB_LIGNES = 100000

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
        
        # Si la classe a moins de NB_LIGNES après le filtrage, on prend le maximum disponible
        # ou on effectue un suréchantillonnage avec replace=True si exigé. 
        # Ici on prend exactement NB_LIGNES si possible pour être strict, quitte à dupliquer (replace=True).
        # Mais en général, s'il y en a assez, replace=False est meilleur.
        nb_dispo = len(temp)
        remplacement = True if nb_dispo < NB_LIGNES else False
        
        echantillon = temp.sample(n=NB_LIGNES, replace=remplacement, random_state=42)
        df_temp = pd.concat([df_temp, echantillon])
        print(f"Note {note} : {NB_LIGNES} avis extraits (Remplacement={'Oui' if remplacement else 'Non'})")
        
    return df_temp

def analyser_categories(df):
    print("\n--- Analyse des catégories du nouvel échantillon ---")
    
    if "categories" in df.columns:
        # On nettoie la colonne souvent formatée comme "['Fiction']"
        categories_top = df["categories"].value_counts().head(10)
        print("Top 10 des catégories les plus représentées :")
        for cat, freq in categories_top.items():
            pourcentage = (freq / len(df)) * 100
            print(f" - {cat} : {freq} avis ({pourcentage:.2f}%)")
    else:
        print("La colonne 'categories' n'est pas présente dans le dataset. Analyse impossible.")
        print("Colonnes disponibles :", df.columns.tolist())

def export_csv(df_a_extraire):
    if not os.path.exists("data"):
        os.mkdir("data")
    
    print("\nMélange des données (Shuffle)...")
    df_a_extraire = df_a_extraire.sample(frac=1, random_state=42).reset_index(drop=True)
    
    nom_fichier = "data/books_rating_500k_filtre.csv"
    df_a_extraire.to_csv(nom_fichier, index=False)
    print(f"\n✅ CSV propre et équidistribué généré avec succès dans '{nom_fichier}'")

if __name__ == "__main__":
    print("Chargement du dataset initial...")
    df = pd.read_csv("data/Books_rating.csv", sep=",")
    
    # 1 et 2. Filtrage (>= 10) et Échantillonnage équidistribué
    df_resultat = extraire_et_filtrer(df)
    
    # 3. Analyse des catégories
    analyser_categories(df_resultat)
    
    # 4. Exportation
    export_csv(df_resultat)