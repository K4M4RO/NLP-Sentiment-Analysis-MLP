# NLP Sentiment Analytics Dashboard

Une approche de Machine Learning pragmatique pour la prédiction de notes (1 à 5 étoiles) d'avis littéraires, basée sur l'analyse sémantique du texte croisée avec les métadonnées de genre.

Le projet repose sur un volume de données significatif (environ **500 000 avis**) et propose une architecture complète : de l'ingestion textuelle à la visualisation de l'espace latent interne du modèle via une application web interactive.

---

## 1. Pipeline Technique & Architecture

Le défi principal de ce projet était de combiner la richesse sémantique du texte libre avec de la donnée tabulaire stricte (le genre du livre).

**L'Encodage et les Features :**
Nous utilisons un embedding textuel dense généré par le transformateur pré-entraîné `all-mpnet-base-v2`, produisant **768 dimensions** sémantiques. S'y ajoute un encodage One-Hot des genres littéraires sur **11 dimensions**. 
La concaténation de ces deux flux offre à notre modèle un espace de représentation complet de **779 features** par avis.

**Le Modèle Prédictif (MLP) :**
Plutôt que d'opter pour des architectures inutilement complexes, nous utilisons un **Perceptron Multicouche (MLPRegressor)**. L'architecture optimale retenue est relativement légère : deux couches cachées concises `(64, 64)`. Cette légèreté étudiée permet de converger rapidement tout en restreignant le nombre de paramètres libres, limitant ainsi la capacité du modèle à "apprendre par cœur" le bruit du dataset.

---

## 2. Réflexion Scientifique et Réalisme

### La Réalité de l'Overfitting
Travailler sur du texte pose toujours le même problème : la très haute dimensionnalité d'un embedding (768 variables pour le sens) crée un espace géométrique où il est mathématiquement extrêmement facile pour un modèle de mémoriser les exemples d'entraînement. L'overfitting n'y est pas un "accident", c'est le comportement naturel du réseau neuronal.

### Notre Résolution par Early Stopping
Pour garantir que notre modèle *comprend* la polarité des notes plutôt que de mémoriser le dataset, nous avons implémenté un contrôle strict de la convergence. À la place de la méthode `.fit()` classique, nous entraînons le réseau de manière séquentielle (Epoch par Epoch via `.partial_fit()`). 

À chaque époque, nous évaluons une **Validation Loss** (Perte sur des données jamais vues par l'algorithme). 
Grâce à un mécanisme d'**Early Stopping avec une patience de 10 époques**, l'entraînement s'arrête net dès que le réseau commence à sur-apprendre sur le dataset principal. Ce protocole nous assure de figer et de sauvegarder la version stricte du modèle qui généralise le mieux (obtenant des performances très correctes : un **F1-Score autour de 0.87**).

---

## 3. Le Dashboard Dash & Cartographie de l'Espace Latent

L'aboutissement du projet est accessible via une interface web propulsée par `Dash`. La pièce maîtresse de ce Dashboard est sa cartographie 2D générée via l'algorithme de réduction de dimensionnalité **UMAP**.

### Le "Twist" Technique de la Cartographie
Généralement, l'UMAP est utilisé pour projeter directement les vecteurs asymétriques issus de BERT. Le problème de cette approche classique est qu'elle regroupe les textes *par thème sémantique* (ex: "les livres de magie ensemble", "les essais politiques ensemble"), indépendamment de l'avis du lecteur.

Pour contourner cela, **notre UMAP ne projette pas la donnée d'entrée**. 
Nous faisons glisser les requêtes au travers du réseau neuronal, et nous extrayons matriciellement les activations générées par la **dernière couche cachée du MLP**. 

### Pourquoi faire ça ?
Cette dernière couche de 64 dimensions représente l'**Espace Latent** : il s'agit littéralement des "croyances" internes de notre algorithme juste avant de donner sa note. Voir cet espace latent à travers UMAP permet de prouver visuellement que notre modèle a réussi à "tordre" l'espace sémantique originel de BERT pour établir une frontière de décision claire et isoler géométriquement les Textes Positifs (Intéressants) des Textes Négatifs (Inintéressants).

---

## 4. Installation et Utilisation

L'architecture s'installe et s'exécute aisément. Recommandation : utiliser un environnement virtuel (Python 3.9+).

**1. Installation des dépendances**
```bash
pip install -r requirements.txt
```

**2. Entraînement du réseau de neurones**
Récupère les représentations pré-calculées et entraîne le MLP avec suivi de la Validation Loss :
```bash
python main.py --train
```

**3. Génération des données pour l'interface**
Calcule les inférences spatiales et l'extraction de l'Espace Latent UMAP :
```bash
python main.py --project
```

**4. Lancement du Dashboard Web**
Démarre le serveur Analytics en local :
```bash
python app.py
```
*Le tableau de bord interactif sera alors disponible sur votre navigateur via `http://127.0.0.1:8050/`.*
