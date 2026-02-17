from composants.ia_notes import NotesPredicteur
import os

if __name__ == "__main__":
    predicteur_notes = NotesPredicteur()
    
    while True:
        # On nettoie la console pour avoir un menu propre à chaque fois
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "═"*50)
        print("       🤖 MENU INTELLIGENCE ARTIFICIELLE")
        print("═"*50 + "\n")
        
        print(" 1 - Charger Données CSV")
        print(" 2 - Calculer Vecteurs (BERT)")
        print(" 3 - Sauvegarder Embeddings (.npy)")
        print(" 4 - Lancer Entraînement")
        print(" 5 - Sauvegarder Cerveau (.joblib)")
        print(" 6 - Charger Cerveau")
        print(" 7 - Charger Embeddings")
        print(" 8 - Prédire une note")
        print(" 9 - Quitter")
        
        print("\n" + "-"*50)
        choix = input("👉 Choisissez une option : ")
        print("-" * 50 + "\n")

        if choix == "1":
            chemin = input("Entrez le chemin du csv (laisser vide pour défaut) : ")
            if chemin.strip() == "":
                predicteur_notes.charger_data()
            else:
                predicteur_notes.charger_data(chemin)

        elif choix == "2":
            predicteur_notes.calculer_vecteurs_semantiques()

        elif choix == "3":
            nom_fichier = input("Nom de fichier (laisser vide pour défaut) : ")
            if nom_fichier.strip() != "":
                predicteur_notes.sauvegarder_embeddings(nom_fichier)
            else:
                predicteur_notes.sauvegarder_embeddings()

        elif choix == "4":
            predicteur_notes.lancer_entrainement()

        elif choix == "5":
            nom_fichier = input("Nom de fichier (laisser vide pour défaut) : ")
            if nom_fichier.strip() != "":
                predicteur_notes.sauvegarder_cerveau(nom_fichier)
            else:
                predicteur_notes.sauvegarder_cerveau() # Correction : Ajout des ()

        elif choix == "6":
            nom_fichier = input("Nom de fichier (laisser vide pour défaut) : ")
            if nom_fichier.strip() != "":
                predicteur_notes.charger_cerveau(nom_fichier)
            else:
                predicteur_notes.charger_cerveau()

        elif choix == "7":
            nom_fichier = input("Nom de fichier (laisser vide pour défaut) : ")
            if nom_fichier.strip() != "":
                predicteur_notes.charger_embeddings(nom_fichier)
            else:
                predicteur_notes.charger_embeddings()

        elif choix == "8":
            texte = input("📝 Entrez le texte à analyser : ")
            try:
                score = predicteur_notes.predire_score(texte)
                print(f"\n⭐ Score prédit : {score}")
            except Exception as e:
                print(f"\n❌ Erreur : {e}")

        elif choix == "9":
            print("Au revoir !")
            break

        else:
            print("❌ Option invalide (1-9)")
        
        # Pause pour laisser le temps de lire
        input("\n[Appuyez sur Entrée pour continuer...]")