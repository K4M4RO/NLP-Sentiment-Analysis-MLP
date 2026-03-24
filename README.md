# NLP Sentiment Analytics Dashboard

Une architecture de Machine Learning robuste et complète, spécialisée dans l'analyse sémantique et la régression de notes d'avis en langage naturel. Ce projet intègre l'encodage par transformateurs pré-entraînés (BERT), un réseau de neurones profond (Perceptron Multicouche) et une interface web interactive de très haute performance.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?style=for-the-badge&logo=dash&logoColor=white)

---

## Architecture du Projet

Le pipeline de traitement s'articule autour de quatre strates majeures, optimisant la mémoire et la puissance de calcul sur un volume massif de données (500 000 avis).

### 1. Préparation et Nettoyage des Données

La robustesse du modèle prédictif reposant sur la qualité du jeu de données, un pipeline rigoureux a été implémenté via le script utilitaire `fouille_donnees.py`. Le jeu de données initial, issu d'une archive brute massive (environ 3 000 000 de lignes), a subi les traitements mathématiques et logiques suivants :

* **Purge des valeurs manquantes :** Élimination systématique des entrées présentant des valeurs nulles (`NaN`) sur l'axe sémantique (`review/text`) ou quantitatif (`review/score`).
* **Seuillage de consistance statistique :** Filtrage conditionnel garantissant que seuls les ouvrages ou items possédant un corpus minimal strict de 10 avis originaux sont conservés dans le jeu final.
* **Équidistribution stricte (Balancing) :** Afin de prévenir les biais de prédiction structurels (overfitting sur des classes majoritaires), un sous-échantillonnage aléatoire a été imposé. Le pipeline extrait exactement 100 000 avis pour chaque classe de notation (de 1.0 à 5.0).
* **Bilan dimensionnel :** Le jeu de données qualifié passe d'un état originel chaotique à un DataFrame parfaitement équilibré de 500 000 enregistrements. Il est persisté localement sous `books_rating_500k_filtre.csv`, offrant une distribution de labels optimale prête pour la phase de vectorisation contextuelle algorithmique.

### 2. Exécution du Pipeline (Mode CLI)

L'architecture backend (`main.py`) adopte un standard d'interface en ligne de commande (CLI) industriel via le module `argparse`. Cette conception modulaire autorise le partitionnement du temps de calcul selon la phase d'expérimentation en cours.

**Arguments de compilation et de lancement (Flags) :**

* `--pipeline` : Facteur de déclenchement d'un cycle systémique de bout en bout (A à Z). Engage la phase de lecture Big Data, la vectorisation CUDA sous BERT, la persistance matricielle des embeddings, le Model Selection sur réseau neuronal, l'arrêt par Early Stopping, et l'extrusion finale de la projection UMAP.
* `--train` : Flag de re-compilation de la fonction de perte limitant l'exécution à la création d'architectures MLP. Ce paramètre court-circuite le lourd process d'encodage asynchrone BERT en récupérant directement les matrices pré-calculées sur disque (`.npy`).
* `--project` : Flag de dérivation analytique isolant le moteur d'apprentissage (MLP). Dédié iniquement au recalcul brut des matrices de coordonnées UMAP/ACP destinées au Dashboard interactif en cas d'ajustement du voisinage statistique.
* `--predict [TEXTE]` : Option d'inférence à froid. Engage le re-chargement exclusif du modèle optimisé en RAM (`.joblib`) afin d'évaluer instantanément la chaîne de caractères et de retourner sa régression et sa classe de polarité dans les flux standards du terminal.

**Exemples conjoints de cycle d'utilisation :**

Exécuter une refonte totale de l'entraînement des poids du modèle en se reposant sur les dimensions déjà persistées :
```bash
python main.py --train
```

Profiler une chaîne non étiquetée directement depuis l'invite de commande (CLI Inférence) :
```bash
python main.py --predict "The pacing of this novel is undeniably slow, yet intellectually rewarding."
```


### 3. Le Cœur du Réacteur (IA & Web)

* **Vectorisation Sémantique (BERT) :** Passage du texte brut à un espace vectoriel dense via le modèle `all-mpnet-base-v2` (SentenceTransformers).
* **Modélisation Neuronale (MLP) :** Régression supervisée avec sélection d'architecture dynamique (Early Stopping sur architecture multiniveaux `128-64-32`).
* **Déploiement Web (Dash/Plotly) :** Restitution visuelle et fonctionnelle, couplant une capacité d'inférence en temps réel à une topologie sémantique projetée par algorithme UMAP.

---

## Exécution du Pipeline (Logs du Terminal)

L'entraînement du modèle, de par sa nature computationnelle intensive, a été découpé et sécurisé par batches. L'utilisation d'une accélération matérielle (Tensor Cores / CUDA) est particulièrement visible et contrôlée au sein de nos barres de progression système. 

**Encodage Vectoriel par le modèle BERT :**
```text
Encodage BERT:  20%|███████████▏ | 396/1954 [11:01<42:58,  1.65s/batch]
```

**Optimisation du Réseau de Neurones Multicouche :**
```text
Epochs completed:  50%| ████████████████████████████████████▌                                      100/200 [02:03]completed  100  /  200 epochs
```

---

## Test d'Intelligence & Sarcasme

L'usage d'un modèle d'attention (Transformers) confère à notre réseau une compréhension contextuelle profonde de la phrase, surpassant les traditionnelles approches statistiques naïves. 

Afin d'éprouver le réseau au-delà de sa boucle d'apprentissage, nous avons simulé un cas complexe mixant antagonisme et polarité extrême :
* **Input (Texte soumis) :** *"this is a masterpiece of bullshit"*
* **Résultat de l'IA (Note prédite) :** `1.10 / 5.0` (Classification : Inintéressant)

**Explication de la prédiction :** 
Malgré la détection évidente du terme fortement valorisant *"masterpiece"* (chef-d'œuvre), le modèle n'est pas tombé dans le piège de la notation lexicale. Il a pu mesurer précisément la relation d'inversion sémantique infligée par la vulgarité ironique *"bullshit"*, saisissant ainsi le contexte purement sarcastique de la phrase pour la pénaliser et l'aligner à près de 1/5.

---

## Structure du Projet

```text
Projet-S4/
├── app.py                      # Frontend Dash UI & Dashboard interactif
├── main.py                     # CLI industriel (exécution du pipeline / entraînement)
├── requirements.txt            # Dépendances minimalistes pour l'inférence
├── composants/                 # Logique métier & Backend
│   └── ia_notes.py             # Cœur de l'Intelligence Artificielle (Classes, MLP, BERT, UMAP)
└── data/                       # Modèles et configurations pré-calculés
    ├── ia_notes_sauvegarde.joblib   # Cerveau du modèle MLP pré-entraîné
    ├── umap_coords_dash.npy         # Coordonnées 2D figées (UMAP)
    └── umap_y_vrai_dash.npy         # Classes réelles rattachées aux points 2D
```

---

## Le Dashboard Interactif (Web UI)

L'interface analytique Dash a été conçue sur une architecture technique "SaaS", axée sur trois panneaux opérationnels sans rechargement de page.

1. **Interface de Test :** Une ligne de commande de texte dynamique permettant l'inférence de manière interactive. Chaque avis saisi passe dans le conduit BERT avant que le Cerveau (MLP joblib) n'émette une évaluation à chaud.
   
   ![Capture Interface de Test](assets/test.png)

2. **Cartographie Sémantique UMAP :** Un graphique Plotly restituant l'espace vectoriel d'un panel de documents et la séparabilité des matrices de notes. Parfaitement fluide sur un échantillon optimisé de 3 000 avis.
   
   ![Capture Cartographie UMAP](assets/umap.png)

    Les points réagissent au survol intelligent (Hover) pour dévoiler le contenu textuel exact ayant placé l'avis dans sa grappe vectorielle correspondante.
   ![Capture Rapport Performance](assets/perf2.png)

3. **Rapport de Performance :** Panneau de métriques quantitatives (R² Score sur les données de test, taille du réseau retenu, matrice de confusion finale) et un score technique général F1 pondéré validé à `0.81`.

   ![Capture Rapport Performance](assets/perf.png)

---

## Prérequis Système

- **Python 3.9+** (l'utilisation d'un environnement virtuel est vivement recommandée).

> **⚠️ Avertissement de Data Lifecycle :**
> En raison des limites systématiques de GitHub concernant l'upload de fichiers excédant 100 Mo, le fichier CSV natif de Big Data (`books_rating_500k_filtre.csv`) n'est pas instancié dans ce dépôt public.
> L'onglet et les calculs frontaux de UMAP fonctionneront infailliblement grâce aux projections topologiques allégées du répertoire (`.npy`). Cependant, notez que **l'affichage dynamique des extraits textuels de l'avis lors du survol de la souris (Hover)** exigera formellement la présence débloquée du CSV originel déposé arbitrairement dans votre dossier local `data/`.

---

## Installation et Lancement

1. Installez l'ensemble des dépendances spécifiques via le gestionnaire de paquets :
```bash
pip install -r requirements.txt
```

2. Lancez le serveur local Dash de l'application :
```bash
python app.py
```
*Le tableau de bord sera instantanément accessible depuis votre navigateur via l'adresse standard `http://127.0.0.1:8050/`.*

---
