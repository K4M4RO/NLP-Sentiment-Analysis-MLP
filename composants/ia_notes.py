from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib, os
from sklearn.decomposition import PCA
# torch, sentence_transformers et umap sont importés localement pour accélérer le CLI

class C: # Classe qui permet d'appliquer de la couleur aux print (pas important)
    VERT = '\033[92m'
    ROUGE = '\033[91m'
    JAUNE = '\033[93m'
    BLEU = '\033[94m'
    RESET = '\033[0m' # Important pour arrêter la couleur

class NotesPredicteur:
    _x: list[str]
    _y: list[str]
    _embeddings: list[list[float]]
    _mlp: object
    _scaler: object
    _bert: object
    _dossier_sauvegarde: str
    _score_ia: dict[float]

    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        
        if torch.cuda.is_available():
            # Active les cœurs Tensor Cores pour accélerer (sacrifie un peu la précision)
            torch.set_float32_matmul_precision('high')
            
        self._x = []
        self._y = []
        self._embeddings = []
        self._scaler = StandardScaler()
        self._dossier_sauvegarde = "data"

        # --- DÉTECTION DU GPU ---
        self._materiel_calcul = "cuda" if torch.cuda.is_available() else "cpu"
        self._modele_bert = SentenceTransformer('all-mpnet-base-v2', device=self._materiel_calcul)

        # --- RECHERCHE DE MODÈLE (Initialisé à None) ---
        self._mlp = None

    def charger_data(self,chemin_dataset:str="data/books_rating_500k_filtre.csv") -> None:
        df = pd.read_csv(chemin_dataset, sep=",", on_bad_lines='skip')
        df = df.dropna(subset=["review/text", "review/score"])
        
        # --- AJOUT : NETTOYAGE STRICT ---
        # On s'assure que tout est bien en texte
        df["review/text"] = df["review/text"].astype(str)
        
        self._y = df["review/score"].to_numpy()
        self._y = np.array(self._y).reshape(-1, 1)
        self._x = df["review/text"].tolist()
        
        print(f"\n{C.BLEU}Données chargées : {len(self._x)} lignes.{C.RESET}")
        
        # --- AJOUT : VÉRIFICATION VISUELLE ---
        print(f"{C.JAUNE}--- VÉRIFICATION ÉCHANTILLON ---{C.RESET}")
        for i in range(3):
            print(f"Note: {self._y[i][0]} | Texte: {self._x[i][:50]}...")
        print(f"{C.JAUNE}----------------------------------{C.RESET}")
        print("Si un texte positif a une note de 1.0 (ou inversement), le fichier CSV est faux !")
    
    def sauvegarder_embeddings(self, nom_fichier="embeddings.npy"):
        # On sauvegarde le vecteur sémantique car il est long à recalculer

        if not os.path.exists(self._dossier_sauvegarde):
            os.makedirs(self._dossier_sauvegarde)
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        np.save(chemin,self._embeddings)
        print(f"\n{C.VERT}Embeddings sauvegardé dans {nom_fichier}!{C.RESET}")
    
    def charger_embeddings(self, nom_fichier="embeddings.npy"):
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        try:
            self._embeddings = np.load(chemin)
            print(f"\n{C.VERT}Embeddings chargé depuis {nom_fichier}!{C.RESET}")
        except:
            print(f"\n{C.ROUGE}Erreur lors du chargement du embeddings !{C.RESET}")
        
    def sauvegarder_cerveau(self, nom_fichier="ia_notes_sauvegarde.joblib"):
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        package = {
            "mlp": self._mlp,
            "scaler": self._scaler,
            "score_ia": self._score_ia
        }
        joblib.dump(package,chemin)
        print(f"\n{C.VERT}IA sauvegardée dans {nom_fichier} !{C.RESET}")
    
    def charger_cerveau(self,nom_fichier="ia_notes_sauvegarde.joblib"):
        chemin = os.path.join(self._dossier_sauvegarde,nom_fichier)
        try:
            package = joblib.load(chemin)
            self._mlp = package["mlp"]
            self._scaler = package["scaler"]
            self._score_ia = package["score_ia"]
            print(f"\n{C.VERT}Cerveau chargé !{C.RESET}")
        except:
            print(f"\n{C.ROUGE}Erreur lors du chargement. Vérifiez que le fichier .joblib existe et qu'il se trouve à la racine{C.RESET}")
    
    def calculer_vecteurs_semantiques(self, texte_a_encoder=None) -> None:
        import torch
        from tqdm import tqdm
        import numpy as np
        
        with torch.no_grad():
            if texte_a_encoder is None:
                taille_batch = 256 if self._materiel_calcul == "cuda" else 64
                print(f"\n{C.JAUNE}Encodage de {len(self._x)} textes en vecteurs BERT (Batch: {taille_batch})...{C.RESET}")
                
                tous_les_embeddings = []
                # Découpage par batches et affichage de la progression globale
                for i in tqdm(range(0, len(self._x), taille_batch), desc="Encodage BERT", unit="batch"):
                    batch_textes = self._x[i:i + taille_batch]
                    # show_progress_bar=False ici car on utilise notre propre tqdm au-dessus
                    batch_embeddings = self._modele_bert.encode(batch_textes, convert_to_numpy=True, show_progress_bar=False)
                    tous_les_embeddings.append(batch_embeddings)
                
                self._embeddings = np.vstack(tous_les_embeddings)
            else:
                # Si c'est un texte seul, on le met dans une liste car BERT veut un itérable
                return self._modele_bert.encode([texte_a_encoder], convert_to_numpy=True)

    def rechercher_meilleur_modele(self) -> tuple:
        print(f"\n{C.JAUNE}--- ⚔️ RECHERCHE DU MEILLEUR MODÈLE (EARLY STOPPING) ---{C.RESET}")
        x_train, x_test, y_train, y_test = train_test_split(self._embeddings, self._y, test_size=0.2, random_state=42)
        
        y_train_scaled = self._scaler.fit_transform(y_train)
        y_test_scaled = self._scaler.transform(y_test)
        
        architectures = [(30, 30), (64, 64), (128, 64, 32)]
        meilleur_r2 = -float('inf')
        meilleur_modele = None
        meilleurs_scores = {}
        historique_architectures = []
        
        for arch in architectures:
            print(f"\n{C.BLEU}Entraînement de l'architecture {arch}...{C.RESET}")
            mlp_candidat = MLPRegressor(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                learning_rate_init=0.001,
                early_stopping=True,       # Modèle stoppé avant sur-apprentissage
                validation_fraction=0.1,   # On garde 10% pour l'early stopping implicitement
                verbose=True               # AFFICHAGE DE LA PERTE POUR LE PROFESSEUR
            )
            
            mlp_candidat.fit(x_train, y_train_scaled.ravel())
            
            y_pred_scaled = mlp_candidat.predict(x_test)
            y_pred_reel = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            
            mse = mean_squared_error(y_test, y_pred_reel)
            r2 = r2_score(y_test, y_pred_reel)
            
            couleur = C.VERT if r2 > 0.4 else C.JAUNE
            print(f"   {couleur}Architecture {arch} : R2 = {r2:.4f} | MSE = {mse:.4f} (Arrêté à l'itération {mlp_candidat.n_iter_}){C.RESET}")
            
            historique_architectures.append({
                "Architecture": str(arch),
                "R2_Score": float(r2),
                "MSE": float(mse),
                "Epochs": int(mlp_candidat.n_iter_)
            })
            
            if r2 > meilleur_r2:
                meilleur_r2 = r2
                meilleur_modele = mlp_candidat
                meilleurs_scores = {"R2": r2, "Mean Squared Error": mse}
                
        print(f"\n{C.VERT}🏆 Vainqueur : Architecture {meilleur_modele.hidden_layer_sizes} avec R2 = {meilleur_r2:.4f}{C.RESET}")
        
        # SAUVEGARDE DE L'HISTORIQUE DE BENCHMARK
        import json
        chemin_historique = os.path.join(self._dossier_sauvegarde, "historique_architectures.json")
        try:
            with open(chemin_historique, "w") as f:
                json.dump(historique_architectures, f, indent=4)
            print(f"{C.JAUNE}Historique des architectures sauvegardé localement pour le Dashboard.{C.RESET}")
        except Exception as e:
            print(f"{C.ROUGE}Erreur lors de la sauvegarde de l'historique : {e}{C.RESET}")
            
        self._mlp = meilleur_modele
        self._score_ia = meilleurs_scores
        
        return x_test, y_test

    def lancer_entrainement(self) -> None:
        if self._mlp is None:
            x_test, y_test = self.rechercher_meilleur_modele()
        else:
            _, x_test, _, y_test = train_test_split(self._embeddings, self._y, test_size=0.2, random_state=42)
            
        y_pred_scaled = self._mlp.predict(x_test)
        y_pred_reel = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        print(f"\n{C.BLEU}Performance finale (Régression) :")
        print(f"Mean Squared Error: {self._score_ia['Mean Squared Error']:.4f}")
        print(f"R² Score: {self._score_ia['R2']:.4f}{C.RESET}")

        # --- NOUVEAU : MÉTRIQUES DE CLASSIFICATION (Seuil 2.5) ---
        print(f"\n{C.JAUNE}--- Métriques de Classification du meilleur modèle (Seuil Strict > 2.5) ---{C.RESET}")
        # Binarisation des notes réelles (y_test) et des prédictions (y_pred_reel)
        y_test_bin = (y_test > 2.5).astype(int)
        y_pred_bin = (y_pred_reel > 2.5).astype(int)

        print(f"{C.VERT}Matrice de confusion :{C.RESET}")
        print(confusion_matrix(y_test_bin, y_pred_bin))

        print(f"\n{C.VERT}Rapport de classification :{C.RESET}")
        print(classification_report(y_test_bin, y_pred_bin, target_names=["Inintéressant (<=2.5)", "Intéressant (>2.5)"]))
        # -----------------------------------------------------------

    def predire_score(self, texte_a_predire: str) -> float:
        # 1. On récupère le vecteur du texte unique
        vecteur = self.calculer_vecteurs_semantiques(texte_a_predire)
        
        # 2. On prédit (le résultat est un array normalisé)
        y_pred_scaled = self._mlp.predict(vecteur)
        
        # 3. On inverse la normalisation pour revenir à l'échelle 1-5
        # reshape(-1, 1) est nécessaire car le scaler veut du 2D
        note_reelle = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # On retourne juste le nombre
        return float(note_reelle[0][0])
    
    def calculer_projections(self, nom_fichier_pca="pca_coords.npy", nom_fichier_umap="umap_coords_dash.npy"):
        if len(self._embeddings) == 0:
            print(f"{C.ROUGE}Erreur : Veuillez calculer ou charger les embeddings d'abord (Option 2 ou 7).{C.RESET}")
            return
            
        print(f"\n{C.BLEU}Standardisation des embeddings pour la projection...{C.RESET}")
        scaler_proj = StandardScaler()
        embeddings_scaled = scaler_proj.fit_transform(self._embeddings)
        
        print(f"{C.BLEU}Calcul de l'ACP (2 composantes)...{C.RESET}")
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(embeddings_scaled)
        
        if not os.path.exists(self._dossier_sauvegarde):
            os.makedirs(self._dossier_sauvegarde)
            
        chemin_pca = os.path.join(self._dossier_sauvegarde, nom_fichier_pca)
        np.save(chemin_pca, pca_coords)
        print(f"{C.VERT}✅ Coordonnées ACP sauvegardées dans {nom_fichier_pca}{C.RESET}")
        variance = pca.explained_variance_ratio_
        print(f"Variance expliquée par l'ACP : {variance[0]:.4f} et {variance[1]:.4f} (Total: {sum(variance):.4f})")
        
        print(f"\n{C.JAUNE}Calcul de UMAP sur la totalité des données ({len(embeddings_scaled)} embeddings)...{C.RESET}")
        print(f"{C.JAUNE}⏳ Attention : Ce calcul UMAP (metric='cosine') peut prendre 20 à 30 minutes.{C.RESET}")
        
        # On calcule UMAP sur TOUTE la donnée (hyper-performant sur GPU, mais très long sur base CPU)
        # verbose=True permet au professeur de voir la barre de progression
        import umap.umap_ as umap
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, n_jobs=-1, verbose=True)
        umap_coords_full = umap_model.fit_transform(embeddings_scaled)
        
        print(f"\n{C.VERT}✅ UMAP terminé. Sous-échantillonnage pour Dash en cours...{C.RESET}")
        
        # Sous-échantillonnage pour Dash (10 000 points maximum pour fluidité)
        nb_samples_dash = min(10000, len(umap_coords_full))
        indices_dash = np.random.choice(len(umap_coords_full), nb_samples_dash, replace=False)
        
        umap_coords_dash = umap_coords_full[indices_dash]
        y_dash = self._y[indices_dash]
        
        chemin_umap = os.path.join(self._dossier_sauvegarde, nom_fichier_umap)
        np.save(chemin_umap, umap_coords_dash)
        
        # On sauvegarde aussi les vraies notes associées pour l'affichage Dash
        chemin_y_umap = os.path.join(self._dossier_sauvegarde, "umap_y_vrai_dash.npy")
        np.save(chemin_y_umap, y_dash)
        
        print(f"{C.VERT}✅ Coordonnées UMAP (échantillon optimisé de {nb_samples_dash} points) sauvegardées pour Dash !{C.RESET}")
    
    def __str__(self) -> str:
        texte = ""

        for element, resultat in self._score_ia.items():
            texte = texte + f"{element}: {resultat}"
        return texte