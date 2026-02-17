import torch
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib, os

if torch.cuda.is_available():
    # Active les cœurs Tensor Cores pour aller beaucoup plus vite (léger sacrifice de précision négligeable)
    torch.set_float32_matmul_precision('high')

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
    _scaler:object
    _bert:object
    _dossier_sauvegarde:str

    def __init__(self):
        self._x = []
        self._y = []
        self._embeddings = []
        self._scaler = StandardScaler()
        self._dossier_sauvegarde = "data"

        # --- DÉTECTION DU GPU ---
        self._materiel_calcul = "cuda" if torch.cuda.is_available() else "cpu"
        self._modele_bert = SentenceTransformer('all-MiniLM-L6-v2', device=self._materiel_calcul)

        # Définition du MLP
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, 64),  # 2 couches cachées de 64 neurones
            activation='relu',            # Fonction d'activation ReLU
            solver='adam',                # Optimiseur Adam
            max_iter=500,                 # Nombre maximal d'itérations
            random_state=42,              # Pour la reproductibilité
            learning_rate_init=0.001,     # Taux d'apprentissage initial
            verbose=True                  # Permet de voir l'évolution de l'erreur à chaque itération
        )

    def charger_data(self,chemin_dataset:str="data/books_rating_500k.csv") -> None:
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
            "scaler": self._scaler
        }
        joblib.dump(package,chemin)
        print(f"\n{C.VERT}IA sauvegardée dans {nom_fichier} !{C.RESET}")
    
    def charger_cerveau(self,nom_fichier="ia_notes_sauvegarde.joblib"):
        chemin = os.path.join(self._dossier_sauvegarde,nom_fichier)
        try:
            package = joblib.load(chemin)
            self._mlp = package["mlp"]
            self._scaler = package["scaler"]
            print(f"\n{C.VERT}Cerveau chargé !{C.RESET}")
        except:
            print(f"\n{C.ROUGE}Erreur lors du chargement. Vérifiez que le fichier .joblib existe et qu'il se trouve à la racine{C.RESET}")
    
    def calculer_vecteurs_semantiques(self, texte_a_encoder=None) -> None:
        with torch.no_grad():
            if texte_a_encoder is None:
                taille_batch = 256 if self._materiel_calcul == "cuda" else 32
                self._embeddings = self._modele_bert.encode(self._x, convert_to_numpy=True, show_progress_bar=True, batch_size=taille_batch)
            else:
                # Si c'est un texte seul, on le met dans une liste car BERT veut un itérable
                return self._modele_bert.encode([texte_a_encoder], convert_to_numpy=True)

    def lancer_entrainement(self) -> None:
        x_train, x_test, y_train, y_test = train_test_split(self._embeddings, self._y, test_size=0.2, random_state=42)
        
        y_train_scaled = self._scaler.fit_transform(y_train)
        y_test_scaled = self._scaler.transform(y_test)

        self._mlp.fit(x_train, y_train_scaled.ravel())
        
        y_pred_scaled = self._mlp.predict(x_test)
        y_pred_reel = self._scaler.inverse_transform(y_pred_scaled.reshape(-1,1))
        mse = mean_squared_error(y_test, y_pred_reel)
        r2 = r2_score(y_test, y_pred_reel)

        print(f"\n{C.BLEU}Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}{C.RESET}")

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
