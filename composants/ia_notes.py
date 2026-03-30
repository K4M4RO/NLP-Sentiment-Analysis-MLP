from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib, os, json
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
    _genres: np.ndarray          # Matrice One-Hot des genres (int8)
    _cols_genre: list[str]       # Noms des colonnes genre_ (ordre = référence)
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
        self._genres = np.array([])     # Initialisé vide
        self._cols_genre = []           # Initialisé vide
        self._scaler = StandardScaler()
        self._dossier_sauvegarde = "data"

        # --- DÉTECTION DU GPU ---
        self._materiel_calcul = "cuda" if torch.cuda.is_available() else "cpu"
        self._modele_bert = SentenceTransformer('all-mpnet-base-v2', device=self._materiel_calcul)

        # --- RECHERCHE DE MODÈLE (Initialisé à None) ---
        self._mlp = None

    # ========================================================================
    # CHARGEMENT DES DONNÉES CSV
    # ========================================================================
    def charger_data(self, chemin_dataset: str = "data/books_rating_500k_filtre.csv") -> None:
        df = pd.read_csv(chemin_dataset, sep=",", on_bad_lines='skip')
        df = df.dropna(subset=["review/text", "review/score"])
        
        # --- NETTOYAGE STRICT : texte bien en str ---
        df["review/text"] = df["review/text"].astype(str)
        
        self._y = df["review/score"].to_numpy()
        self._y = np.array(self._y).reshape(-1, 1)
        self._x = df["review/text"].tolist()
        
        # --- EXTRACTION DYNAMIQUE DES COLONNES GENRE ---
        # On détecte toutes les colonnes qui commencent par "genre_"
        cols_genre = [c for c in df.columns if c.startswith("genre_")]
        
        if cols_genre:
            # Extraction sous forme de matrice NumPy (int8 pour économiser la RAM)
            self._genres = df[cols_genre].to_numpy(dtype=np.int8)
            self._cols_genre = cols_genre
            print(f"\n{C.BLEU}Genres détectés : {len(cols_genre)} colonnes One-Hot extraites.{C.RESET}")
            print(f"  → Colonnes : {cols_genre}")
            print(f"  → Forme de la matrice genres : {self._genres.shape}")
        else:
            # Aucune colonne genre trouvée (ancien CSV sans fusion)
            self._genres = np.array([])
            self._cols_genre = []
            print(f"\n{C.JAUNE}Aucune colonne genre_ détectée. Entraînement sur BERT uniquement.{C.RESET}")
        
        print(f"\n{C.BLEU}Données chargées : {len(self._x)} lignes.{C.RESET}")
        
        # --- VÉRIFICATION VISUELLE ---
        print(f"{C.JAUNE}--- VÉRIFICATION ÉCHANTILLON ---{C.RESET}")
        for i in range(3):
            genre_info = ""
            if len(self._cols_genre) > 0:
                idx_actif = np.argmax(self._genres[i])
                genre_info = f" | Genre: {self._cols_genre[idx_actif]}"
            print(f"Note: {self._y[i][0]} | Texte: {self._x[i][:50]}...{genre_info}")
        print(f"{C.JAUNE}----------------------------------{C.RESET}")
        print("Si un texte positif a une note de 1.0 (ou inversement), le fichier CSV est faux !")

    # ========================================================================
    # CONSTRUCTION DU TENSEUR D'ENTRÉE COMBINÉ (BERT + GENRES)
    # ========================================================================
    def _construire_x_combined(self) -> np.ndarray:
        """
        Concatène horizontalement les embeddings BERT (768 dims) avec la matrice
        One-Hot des genres (N dims). Si aucun genre n'est disponible, retourne
        les embeddings seuls.
        
        Retourne :
            np.ndarray de forme [nb_samples, 768 + N]
        """
        if len(self._genres) > 0 and self._genres.shape[0] == self._embeddings.shape[0]:
            X_combined = np.hstack([self._embeddings, self._genres.astype(np.float32)])
            print(f"{C.BLEU}Tenseur combiné : BERT ({self._embeddings.shape[1]}) + Genres ({self._genres.shape[1]}) = {X_combined.shape[1]} features{C.RESET}")
            return X_combined
        else:
            print(f"{C.JAUNE}Pas de genres disponibles. Utilisation des embeddings BERT seuls ({self._embeddings.shape[1]} features).{C.RESET}")
            return self._embeddings

    # ========================================================================
    # SAUVEGARDE / CHARGEMENT EMBEDDINGS
    # ========================================================================
    def sauvegarder_embeddings(self, nom_fichier="embeddings.npy"):
        if not os.path.exists(self._dossier_sauvegarde):
            os.makedirs(self._dossier_sauvegarde)
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        np.save(chemin, self._embeddings)
        print(f"\n{C.VERT}Embeddings sauvegardé dans {nom_fichier}!{C.RESET}")
    
    def charger_embeddings(self, nom_fichier="embeddings.npy"):
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        try:
            self._embeddings = np.load(chemin)
            print(f"\n{C.VERT}Embeddings chargé depuis {nom_fichier}!{C.RESET}")
        except:
            print(f"\n{C.ROUGE}Erreur lors du chargement du embeddings !{C.RESET}")

    # ========================================================================
    # SAUVEGARDE / CHARGEMENT DU CERVEAU (MODÈLE + CONFIG GENRES)
    # ========================================================================
    def sauvegarder_cerveau(self, nom_fichier="ia_notes_sauvegarde.joblib"):
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        package = {
            "mlp": self._mlp,
            "scaler": self._scaler,
            "score_ia": self._score_ia
        }
        joblib.dump(package, chemin)
        print(f"\n{C.VERT}IA sauvegardée dans {nom_fichier} !{C.RESET}")
        
        # --- SAUVEGARDE DE LA CONFIGURATION DES GENRES ---
        # Ce fichier JSON sert de référence pour l'inférence : il contient
        # l'ordre exact des colonnes genre_ vu pendant l'entraînement.
        if self._cols_genre:
            chemin_config = os.path.join(self._dossier_sauvegarde, "genres_config.json")
            config = {"colonnes_genre": self._cols_genre}
            with open(chemin_config, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"{C.VERT}Configuration des genres sauvegardée dans genres_config.json ({len(self._cols_genre)} genres){C.RESET}")
    
    def charger_cerveau(self, nom_fichier="ia_notes_sauvegarde.joblib"):
        chemin = os.path.join(self._dossier_sauvegarde, nom_fichier)
        print(f"\n{C.JAUNE}Tentative de chargement du modèle depuis : {chemin}{C.RESET}")
        try:
            package = joblib.load(chemin)
            self._mlp = package["mlp"]
            self._scaler = package["scaler"]
            self._score_ia = package["score_ia"]
            print(f"{C.VERT}Cerveau chargé !{C.RESET}")
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement. Vérifiez que le fichier {chemin} existe.\nDétails: {e}")
        
        # --- CHARGEMENT DE LA CONFIGURATION DES GENRES ---
        chemin_config = os.path.join(self._dossier_sauvegarde, "genres_config.json")
        if os.path.exists(chemin_config):
            with open(chemin_config, "r", encoding="utf-8") as f:
                config = json.load(f)
            self._cols_genre = config["colonnes_genre"]
            print(f"{C.BLEU}Configuration genres chargée : {len(self._cols_genre)} colonnes.{C.RESET}")
        else:
            self._cols_genre = []
            print(f"{C.JAUNE}Aucun fichier genres_config.json trouvé. Prédiction sans genre.{C.RESET}")

    # ========================================================================
    # CALCUL DES VECTEURS SÉMANTIQUES (BERT)
    # ========================================================================
    def calculer_vecteurs_semantiques(self, texte_a_encoder=None) -> None:
        import torch
        from tqdm import tqdm
        import numpy as np
        
        with torch.no_grad():
            if texte_a_encoder is None:
                taille_batch = 256 if self._materiel_calcul == "cuda" else 64
                print(f"\n{C.JAUNE}Encodage de {len(self._x)} textes en vecteurs BERT (Batch: {taille_batch})...{C.RESET}")
                
                tous_les_embeddings = []
                for i in tqdm(range(0, len(self._x), taille_batch), desc="Encodage BERT", unit="batch"):
                    batch_textes = self._x[i:i + taille_batch]
                    batch_embeddings = self._modele_bert.encode(batch_textes, convert_to_numpy=True, show_progress_bar=False)
                    tous_les_embeddings.append(batch_embeddings)
                
                self._embeddings = np.vstack(tous_les_embeddings)
            else:
                # Si c'est un texte seul, on le met dans une liste car BERT veut un itérable
                return self._modele_bert.encode([texte_a_encoder], convert_to_numpy=True)

    # ========================================================================
    # RECHERCHE DU MEILLEUR MODÈLE (MODEL SELECTION + EARLY STOPPING)
    # ========================================================================
    def rechercher_meilleur_modele(self) -> tuple:
        print(f"\n{C.JAUNE}--- ⚔️ RECHERCHE DU MEILLEUR MODÈLE (EARLY STOPPING MANUEL) ---{C.RESET}")
        
        # --- CONSTRUCTION DU TENSEUR COMBINÉ (BERT + GENRES) ---
        X_combined = self._construire_x_combined()
        
        x_train, x_test, y_train, y_test = train_test_split(X_combined, self._y, test_size=0.2, random_state=42)
        
        y_train_scaled = self._scaler.fit_transform(y_train)
        y_test_scaled = self._scaler.transform(y_test)
        
        architectures = [(30, 30), (64, 64), (128, 64, 32)]
        meilleur_r2 = -float('inf')
        meilleur_modele = None
        meilleurs_scores = {}
        historique_architectures = []
        
        for arch in architectures:
            print(f"\n{C.BLEU}Entraînement de l'architecture {arch}...{C.RESET}")
            
            # Splitting Train pour la Validation Interne
            x_tr, x_val, y_tr, y_val_scaled = train_test_split(x_train, y_train_scaled.ravel(), test_size=0.1, random_state=42)
            
            mlp_candidat = MLPRegressor(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                random_state=42,
                learning_rate_init=0.001
            )
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            val_loss_curve = []
            
            from sklearn.metrics import confusion_matrix, classification_report, f1_score
            
            # Boucle d'entraînement manuelle pour capter la validation loss à chaque Epoch
            for epoch in range(500):
                mlp_candidat.partial_fit(x_tr, y_tr)
                
                # Validation Loss au format attendu (identique log loss interne 0.5 * MSE)
                y_val_pred = mlp_candidat.predict(x_val)
                val_loss = 0.5 * mean_squared_error(y_val_scaled, y_val_pred)
                val_loss_curve.append(val_loss)
                
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"   {C.JAUNE}Early stopping (patience atteinte) à l'itération {epoch+1}{C.RESET}")
                    break
                    
            # Injecter val_loss_curve pour qu'elle puisse être récupérée par app.py plus tard
            mlp_candidat.val_loss_curve_ = val_loss_curve
            
            # Évaluation du Test d'architecture
            y_pred_scaled = mlp_candidat.predict(x_test)
            y_pred_reel = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
            
            mse = mean_squared_error(y_test, y_pred_reel)
            r2 = r2_score(y_test, y_pred_reel)
            
            # Métriques de Classification pour Dashboard
            y_test_bin = (y_test > 2.5).astype(int).ravel()
            y_pred_bin = (y_pred_reel > 2.5).astype(int).ravel()
            
            mat_conf = confusion_matrix(y_test_bin, y_pred_bin).tolist()
            f1 = float(f1_score(y_test_bin, y_pred_bin))
            class_report = classification_report(y_test_bin, y_pred_bin, output_dict=True, zero_division=0)
            
            couleur = C.VERT if r2 > 0.4 else C.JAUNE
            print(f"   {couleur}Architecture {arch} : R2 = {r2:.4f} | MSE = {mse:.4f}{C.RESET}")
            
            historique_architectures.append({
                "Architecture": str(arch),
                "R2_Score": float(r2),
                "MSE": float(mse),
                "Epochs": int(len(val_loss_curve)),
                "f1_score": f1,
                "confusion_matrix": mat_conf,
                "classification_report": class_report,
                "val_loss_curve": val_loss_curve
            })
            
            if r2 > meilleur_r2:
                meilleur_r2 = r2
                meilleur_modele = mlp_candidat
                meilleurs_scores = {
                    "R2": r2, 
                    "Mean Squared Error": mse,
                    "f1_score": f1,
                    "confusion_matrix": mat_conf,
                    "classification_report": class_report
                }
                
        print(f"\n{C.VERT}🏆 Vainqueur : Architecture {meilleur_modele.hidden_layer_sizes} avec R2 = {meilleur_r2:.4f}{C.RESET}")
        
        # SAUVEGARDE DE L'HISTORIQUE DE BENCHMARK
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

    # ========================================================================
    # ENTRAÎNEMENT (AVEC MÉTRIQUES)
    # ========================================================================
    def lancer_entrainement(self) -> None:
        if self._mlp is None:
            x_test, y_test = self.rechercher_meilleur_modele()
        else:
            # --- CONSTRUCTION DU TENSEUR COMBINÉ ---
            X_combined = self._construire_x_combined()
            _, x_test, _, y_test = train_test_split(X_combined, self._y, test_size=0.2, random_state=42)
            
        y_pred_scaled = self._mlp.predict(x_test)
        y_pred_reel = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        print(f"\n{C.BLEU}Performance finale (Régression) :")
        print(f"Mean Squared Error: {self._score_ia['Mean Squared Error']:.4f}")
        print(f"R² Score: {self._score_ia['R2']:.4f}{C.RESET}")

        # --- MÉTRIQUES DE CLASSIFICATION (Seuil 2.5) ---
        print(f"\n{C.JAUNE}--- Métriques de Classification du meilleur modèle (Seuil Strict > 2.5) ---{C.RESET}")
        y_test_bin = (y_test > 2.5).astype(int)
        y_pred_bin = (y_pred_reel > 2.5).astype(int)

        print(f"{C.VERT}Matrice de confusion :{C.RESET}")
        print(confusion_matrix(y_test_bin, y_pred_bin))

        print(f"\n{C.VERT}Rapport de classification :{C.RESET}")
        print(classification_report(y_test_bin, y_pred_bin, target_names=["Inintéressant (<=2.5)", "Intéressant (>2.5)"]))

    # ========================================================================
    # PRÉDICTION (INFÉRENCE GENRE-AWARE)
    # ========================================================================
    def predire_score(self, texte_a_predire: str, genre: str = None) -> float:
        """
        Prédit le score d'un texte en mode inférence.
        
        Le vecteur d'entrée est construit en concaténant :
            1. L'embedding BERT du texte (768 dims)
            2. Le vecteur One-Hot du genre (N dims)
        
        RÈGLE ANTI-DISTRIBUTION SHIFT :
            Si `genre` est None, vide, ou inconnu → on active `genre_Inconnu`
            (et non un vecteur de zéros) pour respecter la distribution vue
            pendant l'entraînement (chaque ligne avait toujours exactement un 1).
        
        Args:
            texte_a_predire: Le texte de l'avis à analyser
            genre: Le genre du livre (optionnel). Ex: "Fiction", "Religion"
        
        Returns:
            float: La note prédite entre 1 et 5
        """
        # 1. Calcul du vecteur BERT pour le texte
        vecteur_bert = self.calculer_vecteurs_semantiques(texte_a_predire)
        
        # 2. Construction du vecteur One-Hot genre
        if self._cols_genre:
            vecteur_genre = np.zeros((1, len(self._cols_genre)), dtype=np.float32)
            
            # Déterminer quel genre activer
            genre_cible = None
            if genre and genre.strip():
                # Chercher le genre demandé dans la liste des colonnes connues
                nom_col = f"genre_{genre.strip()}"
                if nom_col in self._cols_genre:
                    genre_cible = nom_col
                else:
                    # Genre inconnu du modèle → fallback sur genre_Inconnu
                    print(f"{C.JAUNE}Genre '{genre}' non reconnu. Fallback → genre_Inconnu.{C.RESET}")
            
            # Si pas de genre spécifié ou genre non reconnu → genre_Inconnu
            if genre_cible is None:
                genre_cible = "genre_Inconnu"
            
            # Activer le bon index (mettre 1 à la bonne position)
            if genre_cible in self._cols_genre:
                idx = self._cols_genre.index(genre_cible)
                vecteur_genre[0, idx] = 1.0
            else:
                # Cas extrême : genre_Inconnu n'existe pas dans la config
                # (ne devrait jamais arriver si fouille_donnees.py est bien exécuté)
                print(f"{C.ROUGE}ATTENTION : genre_Inconnu absent de la config. Activation du premier genre par défaut.{C.RESET}")
                vecteur_genre[0, 0] = 1.0
            
            # 3. Concaténation BERT + Genre
            vecteur_combined = np.hstack([vecteur_bert, vecteur_genre])
        else:
            # Pas de config genre chargée → BERT seul (ancien modèle)
            vecteur_combined = vecteur_bert
        
        # 4. Prédiction via le MLP
        y_pred_scaled = self._mlp.predict(vecteur_combined)
        
        # 5. Inversion de la normalisation pour revenir à l'échelle 1-5
        note_reelle = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        return float(note_reelle[0][0])

    # ========================================================================
    # PROJECTIONS (ACP + UMAP)
    # ========================================================================
    def calculer_projections(self, nom_fichier_pca="pca_coords.npy", nom_fichier_csv="donnees_dashboard_umap.csv"):
        if len(self._embeddings) == 0:
            print(f"{C.ROUGE}Erreur : Veuillez calculer ou charger les embeddings d'abord (Option 2 ou 7).{C.RESET}")
            return
            
        print(f"\n{C.BLEU}Construction du tenseur combiné (BERT + Genres) pour les projections...{C.RESET}")
        X_combined = self._construire_x_combined()
            
        print(f"\n{C.BLEU}Standardisation des embeddings pour la projection PCA...{C.RESET}")
        scaler_proj = StandardScaler()
        X_combined_scaled = scaler_proj.fit_transform(X_combined)
        
        print(f"{C.BLEU}Calcul de l'ACP (2 composantes)...{C.RESET}")
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(X_combined_scaled)
        
        if not os.path.exists(self._dossier_sauvegarde):
            os.makedirs(self._dossier_sauvegarde)
            
        chemin_pca = os.path.join(self._dossier_sauvegarde, nom_fichier_pca)
        np.save(chemin_pca, pca_coords)
        print(f"{C.VERT}✅ Coordonnées ACP sauvegardées dans {nom_fichier_pca}{C.RESET}")
        variance = pca.explained_variance_ratio_
        print(f"Variance expliquée par l'ACP : {variance[0]:.4f} et {variance[1]:.4f} (Total: {sum(variance):.4f})")
        
        nb_samples = 1500
        print(f"\n{C.JAUNE}Sélection d'un échantillon de {nb_samples} points pour UMAP ({X_combined.shape[1]} dimensions)...{C.RESET}")
        actual_samples = min(nb_samples, len(X_combined))
        indices_dash = np.random.choice(len(X_combined), actual_samples, replace=False)
        
        X_sample = X_combined[indices_dash]
        X_sample_scaled = X_combined_scaled[indices_dash]
        
        # =========================================================
        # EXTRACTION DE L'ESPACE LATENT ET PRÉDICTION
        # =========================================================
        if self._mlp is None:
             raise RuntimeError("Aucun modèle MLP n'est chargé. L'export de zéros factices est désormais interdit. "
                                "Veuillez appeler 'predicteur.charger_cerveau()' avant 'calculer_projections()'.")
             
        # Vérification stricte des dimensions
        if X_sample.shape[1] != self._mlp.n_features_in_:
            raise ValueError(f"Erreur Dimension! Le modèle attend {self._mlp.n_features_in_} features, "
                             f"mais a reçu {X_sample.shape[1]} features. Avez-vous entraîné l'IA avec/sans métadonnées de Genre ?")
        
        print(f"{C.BLEU}Extraction vectorielle de l'espace latent du modèle (Hidden Layers)...{C.RESET}")
        
        # 1. Récupération de l'échantillon initial
        X_latent = X_sample.copy()
        
        # 2. Passage manuel à travers les couches cachées (Arrêt avant la couche de sortie)
        for i in range(len(self._mlp.coefs_) - 1):
            # Multiplication matricielle (Poids) + Biais
            X_latent = np.dot(X_latent, self._mlp.coefs_[i]) + self._mlp.intercepts_[i]
            
            # Application de la fonction d'activation
            activation = self._mlp.activation
            if activation == 'relu':
                X_latent = np.maximum(0, X_latent)
            elif activation == 'tanh':
                X_latent = np.tanh(X_latent)
            elif activation == 'logistic':
                from scipy.special import expit
                X_latent = expit(X_latent)
                
        # 3. La dernière couche (prédiction scorée à désécheller) via API officielle pour sécurité
        y_pred_scaled = self._mlp.predict(X_sample)
        
        print(f"{C.JAUNE}Calcul de UMAP sur l'espace latent ({X_latent.shape[1]} dimensions)...{C.RESET}")
        import umap.umap_ as umap
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, verbose=True)
        umap_coords_sample = umap_model.fit_transform(X_latent)
        
        print(f"\n{C.VERT}✅ UMAP sur espace latent terminé.{C.RESET}")
        
        textes_sample = [self._x[i] for i in indices_dash]
        notes_vraies_sample = self._y[indices_dash].ravel()
        
        print(f"{C.BLEU}Mise à l'échelle des prédictions...{C.RESET}")
        notes_predites_sample = self._scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
             
        # Conversion du One-Hot genre vers texte lisible
        genres_sample = []
        if len(self._cols_genre) > 0:
            genres_mat_sample = self._genres[indices_dash]
            for row in genres_mat_sample:
                try:
                    idx_genre = np.argmax(row)
                    nom_genre = self._cols_genre[idx_genre].replace("genre_", "")
                    genres_sample.append(nom_genre)
                except:
                    genres_sample.append("Inconnu")
        else:
            genres_sample = ["Inconnu"] * actual_samples
            
        df_dashboard = pd.DataFrame({
            'UMAP_X': umap_coords_sample[:, 0],
            'UMAP_Y': umap_coords_sample[:, 1],
            'Texte': textes_sample,
            'Note_Vraie': notes_vraies_sample,
            'Note_Predite': notes_predites_sample,
            'Genre': genres_sample
        })
        
        chemin_csv_umap = os.path.join(self._dossier_sauvegarde, nom_fichier_csv)
        df_dashboard.to_csv(chemin_csv_umap, index=False)
        print(f"{C.VERT}✅ Export DataFrame unifié généré : {chemin_csv_umap} ({actual_samples} points){C.RESET}")
    
    def __str__(self) -> str:
        texte = ""

        for element, resultat in self._score_ia.items():
            texte = texte + f"{element}: {resultat}"
        return texte