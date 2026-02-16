from composants.ia_notes import NotesPredicteur

if __name__ == "__main__":
    predicteur_notes = NotesPredicteur()
    #predicteur_notes.charger_data("data/books_data_test.csv")
    """
    predicteur_notes.calculer_vecteurs_semantiques()
    predicteur_notes.sauvegarder_embeddings() # Permet de sauvegarder le vecteur sémantique
    predicteur_notes.lancer_entrainement()
    predicteur_notes.sauvegarder_cerveau() # Permet de sauvegarder le cerveau (mlp et scaler)
    """
    
    """
    Pas besoin de charger le embedding si on charge le cerveau déjà entrainer.
    Mais si on charge un embedding pour entrainer l'ia de nouveau alors il faut
    charger les données en utilisant .charger_data()
    """
    #predicteur_notes.charger_embeddings()  
    #predicteur_notes.lancer_entrainement()
    predicteur_notes.charger_cerveau()

    texte = "i love that book."
    print(f"Score : {predicteur_notes.predire_score(texte)}")
    