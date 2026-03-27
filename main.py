import argparse
import sys
from composants.ia_notes import NotesPredicteur, C

def main():
    parser = argparse.ArgumentParser(
        description="CLI du Projet MLP : Pipeline IA de prédiction de notes."
    )
    
    parser.add_argument("--pipeline", action="store_true", help="Lance l'intégralité du processus de A à Z.")
    parser.add_argument("--train", action="store_true", help="Charge les embeddings et lance la recherche du meilleur modèle.")
    parser.add_argument("--project", action="store_true", help="Charge les embeddings et recalcule uniquement les projections ACP et UMAP.")
    parser.add_argument("--predict", type=str, metavar='"TEXTE"', help="Prédit le score d'une phrase passée entre guillemets.")
    
    # Si aucun argument n'est fourni, on affiche l'aide de argparse et on sort.
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    predicteur = NotesPredicteur()
    
    if args.pipeline:
        print(f"\n{C.VERT} Lancement du Pipeline complet de A à Z...{C.RESET}")
        
        print(f"\n{C.JAUNE}[1/6] Chargement des données CSV...{C.RESET}")
        predicteur.charger_data()
        
        print(f"\n{C.JAUNE}[2/6] Calcul des vecteurs BERT...{C.RESET}")
        predicteur.calculer_vecteurs_semantiques()
        
        print(f"\n{C.JAUNE}[3/6] Sauvegarde des embeddings...{C.RESET}")
        predicteur.sauvegarder_embeddings()
        
        print(f"\n{C.JAUNE}[4/6] Lancement de l'entraînement (Model Selection)...{C.RESET}")
        predicteur.lancer_entrainement()
        
        print(f"\n{C.JAUNE}[5/6] Sauvegarde du Cerveau (meilleur modèle)...{C.RESET}")
        predicteur.sauvegarder_cerveau()
        
        print(f"\n{C.JAUNE}[6/6] Calcul des projections (ACP & UMAP)...{C.RESET}")
        predicteur.calculer_projections()
        
        print(f"\n{C.VERT}🎉 Pipeline terminé avec succès !{C.RESET}")
        
    elif args.train:
        print(f"\n{C.VERT} Lancement du mode Entraînement...{C.RESET}")
        print(f"\n{C.JAUNE}Chargement du CSV (pour récupérer les labels de notes)...{C.RESET}")
        predicteur.charger_data()
        predicteur.charger_embeddings()
        predicteur.lancer_entrainement()
        predicteur.sauvegarder_cerveau()
        print(f"\n{C.VERT}🎉 Entraînement et sauvegarde terminés avec succès !{C.RESET}")
        
    elif args.project:
        print(f"\n{C.VERT} Lancement du mode Projection...{C.RESET}")
        print(f"\n{C.JAUNE}Chargement du CSV (pour cartographier les couleurs des notes)...{C.RESET}")
        predicteur.charger_data()
        predicteur.charger_embeddings()
        predicteur.calculer_projections()
        print(f"\n{C.VERT}🎉 Projections calculées et sauvegardées avec succès !{C.RESET}")
        
    elif args.predict:
        texte = args.predict
        print(f"\n{C.VERT} Lancement du mode Prédiction...{C.RESET}")
        predicteur.charger_cerveau()
        try:
            score = predicteur.predire_score(texte)
            is_interessant = score > 2.5
            texte_resultat = "Intéressant" if is_interessant else "Inintéressant"
            couleur = C.VERT if is_interessant else C.ROUGE
            
            print(f"\n Texte analysé : '{texte}'")
            print(f" Score prédit : {score:.2f}/5")
            print(f" Classification : {couleur}{texte_resultat}{C.RESET}\n")
        except Exception as e:
            print(f"\n{C.ROUGE}❌ Erreur lors de la prédiction : {e}{C.RESET}\n")

if __name__ == "__main__":
    main()