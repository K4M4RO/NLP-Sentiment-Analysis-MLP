# 📚 NotesPredicteur - Analyse d'Avis Amazon

Ce projet implémente un modèle de Machine Learning capable de prédire la note d'un livre (de 1 à 5) à partir du texte d'un avis client. Il utilise des **Embeddings BERT** (`all-MiniLM-L6-v2`) pour la compréhension sémantique et un **Perceptron Multicouche (MLP)** pour la régression.

---

## 🛠️ Prérequis

### Installation Standard (CPU)
Assurez-vous d'avoir installé les bibliothèques suivantes :

```bash
pip install torch sentence-transformers scikit-learn pandas numpy joblib
```

### 🚀 Accélération GPU (NVIDIA RTX) - Optionnel mais Recommandé

Si vous possédez une carte graphique NVIDIA, le calcul des vecteurs sera **beaucoup plus rapide**. Il faut installer une version spécifique de PyTorch compatible avec CUDA.

Exemple pour CUDA 12.8 (commande à adapter selon votre carte graphique) :
```bash
pip3 install torch torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
```

> **Note :** La version `cu128` dépend de vos drivers. Vérifiez la commande exacte correspondant à votre matériel sur le site officiel [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🏗️ Structure du Projet

* `entrainement.py` : Script principal pour piloter l'IA (chargement, entraînement, test).
* `composants/ia_notes.py` : Contient la classe `NotesPredicteur` (la logique métier).
* `data/` : Dossier contenant les fichiers sources (CSV) et les fichiers de sauvegarde générés (`.npy`, `.joblib`).

---

## 🚀 Utilisation de la classe `NotesPredicteur`

La classe est conçue pour être modulaire afin de gagner du temps lors des phases de tests.

### 1. Premier entraînement complet
À utiliser lors de la première manipulation des données. Le calcul des vecteurs (BERT) est l'étape la plus longue.

```python
predicteur = NotesPredicteur()
predicteur.charger_data("data/votre_fichier.csv")
predicteur.calculer_vecteurs_semantiques()  # Étape lente (BERT)
predicteur.sauvegarder_embeddings()        # Sauvegarde le .npy dans /data
predicteur.lancer_entrainement()
predicteur.sauvegarder_cerveau()           # Sauvegarde le .joblib dans /data
```

### 2. Ré-entraînement rapide
Pour tester différentes architectures de neurones (MLP) sans attendre que BERT recalcule tout.

```python
predicteur = NotesPredicteur()
predicteur.charger_data("data/votre_fichier.csv") # Nécessaire pour les labels Y
predicteur.charger_embeddings()                  # Charge le .npy
predicteur.lancer_entrainement()
```

### 3. Prédiction directe (Production)
Mode ultra-rapide pour une utilisation réelle. Ne nécessite que le fichier de sauvegarde final.

```python
predicteur = NotesPredicteur()
predicteur.charger_cerveau() # Charge le .joblib (MLP + Scaler)
note = predicteur.predire_score("This book is amazing, I loved the plot twists!")
print(f"Note estimée : {note:.2f}/5")
```

---

## 📖 Documentation des Méthodes

| Méthode | Description |
| :--- | :--- |
| `charger_data(chemin)` | Charge le CSV et prépare les variables `_x` (textes) et `_y` (notes). |
| `calculer_vecteurs_semantiques()` | Transforme le texte en vecteurs numériques via BERT. |
| `sauvegarder_embeddings(nom)` | Sauvegarde les vecteurs calculés dans un fichier `.npy` pour gagner du temps. |
| `charger_embeddings(nom)` | Charge les vecteurs existants depuis un fichier `.npy`. |
| `lancer_entrainement()` | Divise les données, normalise les cibles et entraîne le MLP. |
| `sauvegarder_cerveau(nom)` | Enregistre le modèle et le scaler dans un fichier `.joblib`. |
| `charger_cerveau(nom)` | Charge un modèle pré-entraîné pour faire des prédictions immédiates. |
| `predire_score(texte)` | Analyse un texte et renvoie une note flottante entre 1.0 et 5.0. |

---

## 📊 Performances (Entraînement sur 500k lignes)

* **Temps d'encodage BERT :** *À compléter (ex: 4h sur CPU)*
* **Temps d'entraînement MLP :** *À compléter (ex: 10 min)*
* **Score R² (Précision) :** *À compléter (ex: 0.82)*

---

## ⚠️ Notes Importantes

> [!IMPORTANT]
> **Gestion des fichiers lourds :** Les fichiers présents dans `data/` (`.csv`, `.npy`, `.joblib`) sont exclus du suivi Git via le fichier `.gitignore` pour éviter de saturer le dépôt distant.

> [!TIP]
> **Normalisation :** Le `StandardScaler` est stocké dans le "Cerveau" (`.joblib`). Il est essentiel pour convertir les valeurs mathématiques du MLP en notes réelles.
