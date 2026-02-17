import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
FICHIER_CIBLE = "data/books_rating.csv"  # Ton fichier extrait

def analyser_donnees():
    # 1. Vérification de l'existence du fichier
    if not os.path.exists(FICHIER_CIBLE):
        print(f"❌ Erreur : Le fichier '{FICHIER_CIBLE}' est introuvable.")
        return

    print(f"📂 Analyse du fichier : {FICHIER_CIBLE}")
    print("-" * 50)

    # 2. Affichage des types de données (Data Types)
    # On charge juste 5 lignes pour voir la structure sans saturer la RAM
    df_preview = pd.read_csv(FICHIER_CIBLE, nrows=5)
    
    print("📊 TYPES DE DONNÉES DÉTECTÉS :")
    print(df_preview.dtypes)
    print("-" * 50)

    # 3. Chargement des notes pour le graphique
    print("⏳ Chargement des notes pour le graphique...")
    try:
        # On ne charge QUE la colonne score pour aller vite
        df = pd.read_csv(FICHIER_CIBLE, usecols=["review/score"])
    except ValueError:
        # Si la colonne s'appelle autrement (ex: 'score' tout court)
        print("⚠️ Colonne 'review/score' non trouvée, tentative de chargement complet...")
        df = pd.read_csv(FICHIER_CIBLE)

    # 4. Comptage des notes
    distribution = df["review/score"].value_counts().sort_index()
    
    print("\n🔢 RÉPARTITION DES NOTES :")
    print(distribution)

    # 5. Création du Diagramme en Barres
    plt.figure(figsize=(10, 6))
    
    # Création des barres
    barres = plt.bar(distribution.index, distribution.values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'])
    
    # Esthétique
    plt.title('Distribution des Notes (1 à 5 étoiles)', fontsize=16)
    plt.xlabel('Note attribuée', fontsize=12)
    plt.ylabel("Nombre d'avis", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks([1, 2, 3, 4, 5]) # Force l'affichage des entiers 1, 2, 3...

    # Ajout du nombre exact au-dessus de chaque barre
    for barre in barres:
        hauteur = barre.get_height()
        plt.text(barre.get_x() + barre.get_width()/2., hauteur,
                 f'{int(hauteur)}',
                 hauteur=0, va='bottom', ha='center', fontsize=10, fontweight='bold')

    print("\n✅ Affichage du graphique...")
    plt.show()

if __name__ == "__main__":
    analyser_donnees()