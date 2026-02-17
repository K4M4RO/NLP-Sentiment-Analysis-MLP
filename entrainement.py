from composants.ia_notes import NotesPredicteur
import os

if __name__ == "__main__":
    """
    Pas besoin de charger le embedding si on charge le cerveau déjà entrainer.
    Mais si on charge un embedding pour entrainer l'ia de nouveau alors il faut
    charger les données en utilisant .charger_data() avant.
    """
    predicteur_notes = NotesPredicteur()
    os.system('cls' if os.name == 'nt' else 'clear')
    while True:
        print("1 - charger_data")
        print("2 - calculer_vecteurs_semantiques")
        print("3 - sauvegarder embeddings")
        print("4 - lancer entrainement")
        print("5 - sauvegarder cerveau")
        print("6 - charger cerveau")
        print("7 - charger embeddings")
        print("8 - predire note")

        choix = input("Choisissez une option :")
        if choix == "1":
            chemin = input("Entrez le chemin du csv (data/books_data.csv) :")
            predicteur_notes.charger_data()
        if choix == "2":
            predicteur_notes.calculer_vecteurs_semantiques()
        if choix == "3":
            nom_fichier = input("Choisissez un nom de fichier (laisser vide pour le choix par défaut) :")
            if nom_fichier != "":
                predicteur_notes.sauvegarder_embeddings(nom_fichier)
            predicteur_notes.sauvegarder_embeddings()
        if choix == "4":
            predicteur_notes.lancer_entrainement()
        if choix == "5":
            nom_fichier = input("Choisissez un nom de fichier (laisser vide pour le choix par défaut) :")
            if nom_fichier != "":
                predicteur_notes.sauvegarder_cerveau(nom_fichier)
            predicteur_notes.sauvegarder_cerveau
        if choix == "6":
            predicteur_notes.charger_cerveau()
        if choix == "7":
            predicteur_notes.charger_embeddings()
        if choix == "8":
            texte = input("Entrez le texte dont vous voulez prédire le score :")
            print(predicteur_notes.predire_score(texte))
        else:
            print("Entrez le nombre de la fonction à executer (1-8)")

    