# NLP Sentiment Analytics Dashboard

Un projet de Machine Learning visant à prédire la note (1 à 5 étoiles) d'un avis littéraire à partir de son texte et de ses métadonnées. L'objectif est d'explorer les techniques de NLP sur un corpus réel (environ 500 000 avis).

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?style=for-the-badge&logo=dash&logoColor=white)

---

## 1. Vue d'Ensemble de l'Architecture

Le pipeline de traitement s'articule autour de l'ingestion des données, de l'entraînement du modèle prédictif et de son analyse visuelle.

### 1.1. Structure du Projet

L'arborescence illustre la séparation stricte entre le front-end asynchrone, les scripts d'exécution et les données compilées.

```text
Projet-S4/
├── app.py                        # Frontend Dash UI & Dashboard interactif
├── main.py                       # Script principal (entraînement, projection, pipeline)
├── fouille_donnees.py            # Script de nettoyage, équilibrage et échantillonnage (ETL)
├── requirements.txt              # Dépendances Python
├── .gitignore                    # Règles d'exclusion Git (gestion des gros fichiers CSV)
├── assets/                       # Dossier contenant les captures d'écran du README
├── composants/                   # Scripts et modules métiers
│   └── ia_notes.py               # Cœur de l'IA : pipeline NLP, MLP, et extraction de l'Espace Latent
└── data/                         # Modèles, caches et données pré-calculées
    ├── Books_rating.csv              # Archive brute des textes Amazon (2.8 Go - source)
    ├── books_data.csv                # Archive brute des métadonnées (181 Mo - titres & genres)
    ├── books_rating_500k_filtre.csv  # Dataset source nettoyé (non tracké via Git LFS)
    ├── donnees_dashboard_umap.csv    # Coordonnées finales et notes pour la cartographie Dash
    ├── embeddings.npy                # Cache des vecteurs sémantiques BERT (768D)
    ├── genres_config.json            # Configuration de l'encodage One-Hot des genres (11D)
    ├── historique_architectures.json # Suivi des performances et du F1-Score des modèles testés
    ├── ia_notes_sauvegarde.joblib    # Poids et biais du modèle MLP optimal sauvegardé
    ├── pca_coords.npy                # Coordonnées de la réduction intermédiaire ACP
    ├── umap_coords_dash.npy          # Coordonnées 2D UMAP historiques
    └── umap_y_vrai_dash.npy          # Classes réelles rattachées aux points 2D
```

---

## 2. Le Pipeline Data & IA (Backend)

Nous avons mis en place un processus d'Extraction et de Transformation simple pour alimenter le modèle.

### 2.1. Préparation et Nettoyage des Données

Le jeu de données initial contient environ 3 millions de lignes. Le script utilitaire `fouille_donnees.py` applique un pipeline de traitement rigoureux :

* **Nettoyage :** Suppression des entrées présentant des valeurs manquantes (NaN) dans le texte de l'avis ou dans le score.
* **Seuillage :** Conservation exclusive des livres ayant au moins 10 avis pour assurer un minimum de consistance statistique.
* **Filtrage par Genre :** Pour intégrer la dimension thématique de manière pertinente, nous avons identifié et isolé les **10 genres littéraires les plus représentés**. Tous les autres genres minoritaires ont été regroupés sous une catégorie "Autre" (soit 11 catégories au total encodées en One-Hot).
* **Équilibrage strict :** Afin de prévenir tout biais d'apprentissage vers les notes habituellement majoritaires (ex: les 5 étoiles), un sous-échantillonnage a été imposé. Nous avons extrait exactement 100 000 avis pour chaque note (de 1.0 à 5.0).

Le jeu de données final, parfaitement équilibré et exploité pour l'entraînement, s'élève à **500 000 enregistrements**.

### 2.2. Mode CLI : Exécution Modulaire (`main.py`)

Les opérations sont pilotées en ligne de commande (CLI) via le module `argparse`.

* `--pipeline` : Paramètre de déclenchement du cycle de traitement complet. Engage la vectorisation sous BERT, la persistance matricielle des embeddings, l'entraînement du réseau neuronal et l'extraction finale de la projection UMAP.
* `--train` : Lance uniquement l'entraînement du modèle MLP. Ce paramètre court-circuite le process d'encodage BERT en récupérant directement les matrices de features pré-calculées (`.npy`).
* `--project` : Utilise le réseau neuronal chargé pour recalculer les coordonnées UMAP destinées au Dashboard interactif.
* `--predict [TEXTE]` : Lance le modèle sauvegardé pour évaluer instantanément une chaîne de caractères et retourner sa prédiction de score dans le terminal.

```bash
# Exemple de commande d'Inférence Terminale :
python main.py --predict "The pacing of this novel is undeniably slow, yet intellectually rewarding."
```

### 2.3. Logs d'Exécution et Accélération Matérielle

L'avancement du calcul des vecteurs et de l'entraînement peut être surveillé en direct dans la console.

**Moteur Tensoriel NLP (BERT) :**
```text
Encodage BERT:  20%|███████████▏ | 396/1954 [11:01<42:58,  1.65s/batch]
```

**Apprentissage (MLP) :**
```text
Epochs completed:  50%| ████████████████████████████████████▌                                      100/200 [02:03]completed  100  /  200 epochs 
```

![Capture du terminal](assets/cli1.png)

![Capture du terminal](assets/cli2.png)

![Capture du terminal](assets/cli3.png)

---

## 3. Benchmark & Modélisation

### 3.1. Approche Technique : Encodage et Réseau Neuronal

La modélisation finale repose sur une combinaison d'embeddings de transformateurs et de réseaux simples.

**L'encodage :** Nous utilisons l'outil `all-mpnet-base-v2` (`sentence-transformers`) pour distiller un vecteur sémantique de **768 dimensions**. Nous y concaténons une information fondamentale : le genre du livre, appliqué via un encodage One-Hot sur **11 dimensions**. Le modèle reçoit donc en entrée **779 features**.

**Le Modèle et l'Architecture :** Dans une démarche empirique, nous avons testé et comparé le F1-Score et le R² de plusieurs architectures de réseaux de neurones (notamment `(30, 30)`, `(64, 64)`, et `(128, 64, 32)`). Le choix s'est porté sur le Perceptron Multicouche (MLP) comportant trois couches cachées : `(128, 64, 32)`. Cette architecture s'est avérée être le meilleur compromis : elle offre les meilleures performances de précision avec un **MSE d'environ 0.6973** et un **$R^2$ de 0.65**, favorisant une mémorisation latente sans sacrifier la généralisation.

### 3.2. Gestion de l'Overfitting

Travailler sur le langage naturel avec un grand nombre de dimensions génère inévitablement de l'overfitting. Le réseau aura tendance, au fil des epochs, à retenir par cœur les données de construction du texte pour minimiser mécaniquement sa perte, au détriment de sa capacité de généralisation.

Nous avons répondu à cela avec pragmatisme : un processus strict d'**Early Stopping** a été codé. L'entraînement est itératif (via `.partial_fit()`) et traque une perte sur des données laissées hors-chantier (Validation Loss). Grâce à une patience de 10 époques (le réseau s'arrête de s'entraîner si la `val_loss` stagne pendant 10 itérations), nous garantissons que le modèle sauvegardé préserve ses capacités de généralisation.

**Restauration des Poids (Restore Best Weights) :**
Pour garantir une généralisation optimale, notre Early Stopping ne se contente pas d'arrêter l'entraînement. Il utilise une sauvegarde profonde (`deepcopy`) pour restaurer automatiquement les meilleurs poids du réseau identifiés *avant* la phase de sur-apprentissage (overfitting). Cela garantit que le modèle final est celui qui possède la plus petite erreur de validation possible.

### 3.3. Protocole d'Évaluation et Intégrité des Données

Le projet applique une séparation stricte du jeu de données initial (500 000 avis) pour garantir la validité scientifique des mesures de performance :

**Répartition Train/Test (80/20) :**
- **Set d'Entraînement (400 000 avis) :** Utilisé pour l'ajustement des poids du réseau de neurones. Une fraction interne de ce set est dédiée à la validation pour le mécanisme d'Early Stopping.
- **Set de Test (100 000 avis) :** Données totalement indépendantes, isolées dès le début du pipeline. Ce set sert exclusivement à l'évaluation finale.

**Provenance des Métriques du Dashboard :**
- Les indicateurs globaux (F1-Score, R², MSE) ainsi que la matrice de confusion certifiée sont calculés sur l'intégralité des 100 000 avis de test.
- L'échantillon de 1 500 points utilisé pour la projection UMAP et la matrice interactive est extrait par tirage aléatoire uniquement au sein de ce set de test.

Cette méthodologie assure que les résultats présentés reflètent la capacité réelle de généralisation du modèle face à des données non rencontrées durant l'apprentissage (absence de Data Leakage).

---

## 4. Le Dashboard Interactif (Web UI)

Pour analyser les résultats, un dashboard Dashboard `app.py` construit avec `Dash` permet de visualiser les outputs sans passer par le terminal.

### 1. Interface de Test Dynamique
Un simple champ texte autorise l'utilisateur à injecter une critique libre. Le passage par le conduit BERT puis l'arborescence MLP pré-entraînée génèrent l'estimation du score sur le champ.

![Capture Interface de Test](assets/test.png)

### 2. Cartographie Sémantique UMAP Interactive

La pièce maîtresse du projet est notre exploitation de la réduction cartographique dimensionnelle UMAP.

Traditionnellement, appliquer un algorithme UMAP sur des données BERT brutes ne fait que structurer les points selon le thème du texte (regroupant les mots par contexte), sans aucun regard sur le sentiment. 
Pour dépasser cette limite, **notre UMAP ne projette pas l'entrée, mais l'Espace Latent du modèle**. Plus techniquement, l'échantillon passe par notre MLP et nous extrayons manuellement les activations associées à la dernière couche cachée (de dimension 64). L'algorithme UMAP est alors appliqué sur cette matrice précise.

Visuellement, cela permet de mesurer comment l'IA a réorganisé les dimensions pour créer une frontière de décision non-linéaire qui sépare de façon cohérente les sentiments (exprimés en notes de 1 à 5).

![Capture Cartographie UMAP](assets/umap.png)

L'interface interagit au survol en affichant l'extrait d'avis unique lié à chaque nœud d'observation.

![Capture Hover Text UMAP](assets/perf2.png)

### 3. Rapport de Forme (Metrics)
Ce panneau dresse formellement le bilan analytique des performances évaluées sur un ensemble de **Test**. En évitant rigoureusement l'overfitting, notre évaluation indépendante retourne un **score F1 robuste d'environ 0.87**, une variabilité capturée de **$R^2$ = 0.65**, et surtout, une erreur quadratique minimisée avec un **MSE d'environ 0.6973**.

![Capture Rapport Performance](assets/perf.png)

---

## 5. Prérequis Système et Déploiement

### 5.1. Avertissement 

> **⚠️ Avertissement (Git LFS) :** 
> Le fichier de travail initial `books_rating_500k_filtre.csv` pèse lourd et ne siège pas sur ce dépôt. Les fonctionnalités de Dashboard interactifs (incluant les calculs Numpy `.npy` de l'UMAP) sont fonctionnelles de base.

- **Prérequis Standard :** L'usage d'une distribution Python `3.9` ou supérieure est recommandé au sein d'un environnement virtuel.

### 5.2. Installation et Lancement Rapide

0. Récupération du projet
```bash
git clone https://github.com/K4M4RO/NLP-Sentiment-Analysis-MLP
```
```bash
cd NLP-Sentiment-Analysis-MLP
```
Environnement virtuel (Recommandé):
```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate
# Activer l'environnement (Mac/Linux)
source venv/bin/activate
```

2. Installez les packages d'inférence (réseau allégé) :
```bash
pip install -r requirements.txt
```

2. (Optionnel) Relancez l'entraînement localement avec suivi de la Validation Loss :
```bash
python main.py --train
```

3. (Optionnel) Générez l'extraction de l'Espace Latent UMAP :
```bash
python main.py --project
```

4. Initiez le serveur Dash :
```bash
python app.py
```
*Le portail applicatif sera accessible via votre port localhost `http://127.0.0.1:8050/`.*
