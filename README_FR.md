# Guide étudiant · Français

**Ting Tan · EEEA · Oct 2026 · L1–L2**

[Accueil](README.md) · [English](README_EN.md)

## 1. Télécharger et ouvrir le cours

1. [Télécharger le pack étudiant](https://github.com/moonlovist/EEEA-ML/raw/refs/heads/main/downloads/ML_Labs_2026_student.zip).
2. Dans l’Explorateur Windows, ouvrir votre dossier personnel et créer `AI_Course`.
3. Faire un clic droit sur le ZIP → **Extraire tout**. Choisir `AI_Course`.
4. Dans le dossier extrait `ML_Labs_2026`, double-cliquer sur **START_HERE.html**.
5. Choisir **Français**, puis **Installer et démarrer**. Le guide explique chaque étape, avec les commandes et les résultats attendus.

Conserver l’ensemble du dossier : les Notebooks utilisent les données et les modules Python qui l’accompagnent. `START_HERE.html` ouvre le portail du cours depuis votre ordinateur. Les pages de lecture, indices et graphiques sont accessibles dès l’extraction.

Sur GitHub, le bouton **Code → Download ZIP** fournit aussi tout le dépôt. Après extraction, ouvrir le sous-dossier `ML_Labs_2026` pour commencer.

## 2. Préparer Python et Jupyter

Suivre le guide Windows du portail. La configuration cible utilise Python 3.13 standard 64 bits sur Windows x64 ; Python 3.11 ou 3.12 convient également au parcours proposé. L’installation initiale demande une connexion ou les paquets préparés par l’enseignant. Prévoir 30 à 45 minutes à la première séance.

Après installation de Python, ouvrir le dossier `ML_Labs_2026` dans l’Explorateur. Cliquer sur la barre d’adresse, saisir `cmd`, puis appuyer sur Entrée. Exécuter les commandes suivantes une par une, en attendant la fin de chaque commande :

```bat
RUN.cmd -m venv .venv
INSTALL_NOTEBOOK.cmd
CHECK_SETUP.cmd
START_NOTEBOOK.cmd
```

`INSTALL_NOTEBOOK.cmd` installe les bibliothèques de calcul et Jupyter. Le diagnostic doit afficher `status: READY`. `START_NOTEBOOK.cmd` démarre Jupyter et ouvre habituellement un onglet du navigateur. Garder le terminal ouvert. Si nécessaire, copier dans le navigateur l’URL locale complète affichée dans le terminal.

Le guide détaillé présente les messages attendus et les solutions aux difficultés d’installation. Les séances suivantes réutilisent l’environnement : double-cliquer sur `START_NOTEBOOK.cmd`.

## 3. Faire le premier exercice

1. Dans le navigateur de fichiers **Jupyter**, ouvrir **notebooks → fr → Lab00_FR.ipynb**.
2. Lire le texte, puis sélectionner la première cellule de code.
3. Prédire la valeur affichée et appuyer sur **Maj+Entrée**. Le résultat apparaît sous la cellule.
4. Continuer dans l’ordre. Le symbole `[*]` indique une exécution en cours.
5. Modifier une valeur selon la consigne, puis relancer la cellule et celles qui en dépendent.
6. Enregistrer avec **Ctrl+S**. Utiliser ensuite **Kernel → Restart Kernel and Run All** pour vérifier l’exécution complète.

Après TP 00, ouvrir TP 01. Chaque TP propose un exemple guidé, un exercice court, deux indices et une modification à comparer. `None` marque une réponse à compléter ; le message affiché vous aide à poursuivre. Utiliser `REPORT_FR.txt` pour rédiger le compte rendu.

## 4. Reconnaître les trois modes de consultation

| Entrée | Utilisation |
| --- | --- |
| Lien `.ipynb` sur GitHub | Lire les cellules dans l’aperçu GitHub |
| Lien Notebook du portail local | Lire la page mise en forme et ouvrir les indices |
| Fichier ouvert dans Jupyter | Modifier, exécuter, afficher les résultats et enregistrer |

Le fichier `.ipynb` stocke les cellules au format JSON. Si un navigateur affiche ce contenu textuel, ouvrir `START_HERE.html` pour lire le TP ou lancer `START_NOTEBOOK.cmd` pour travailler dans Jupyter. Le bouton **Raw** de GitHub affiche le contenu du fichier.

Les résultats pré-calculés se trouvent dans `examples/fr`. Chaque rapport contient les graphiques, les mesures et les versions logicielles utilisées. Les expériences exécutées pendant votre séance écrivent leurs résultats dans `outputs/fr`.

## 5. Progression

| TP / Lab | Français | English |
| --- | --- | --- |
| 00 | [Premiers pas en Python](ML_Labs_2026/notebooks/fr/Lab00_FR.ipynb) | [Python warm-up](ML_Labs_2026/notebooks/en/Lab00_EN.ipynb) |
| 01 | [Données, variables et cibles](ML_Labs_2026/notebooks/fr/Lab01_FR.ipynb) | [Data, features and targets](ML_Labs_2026/notebooks/en/Lab01_EN.ipynb) |
| 02 | [Régression linéaire et erreurs](ML_Labs_2026/notebooks/fr/Lab02_FR.ipynb) | [Linear regression and errors](ML_Labs_2026/notebooks/en/Lab02_EN.ipynb) |
| 03 | [Classification : kNN et arbres](ML_Labs_2026/notebooks/fr/Lab03_FR.ipynb) | [Classification: kNN and trees](ML_Labs_2026/notebooks/en/Lab03_EN.ipynb) |
| 04 | [Validation, surapprentissage et choix du modèle](ML_Labs_2026/notebooks/fr/Lab04_FR.ipynb) | [Validation, overfitting and model selection](ML_Labs_2026/notebooks/en/Lab04_EN.ipynb) |
| 05 | [K-means et standardisation](ML_Labs_2026/notebooks/fr/Lab05_FR.ipynb) | [K-means and scaling](ML_Labs_2026/notebooks/en/Lab05_EN.ipynb) |
| 06 | [Réseau de neurones, gradients et apprentissage](ML_Labs_2026/notebooks/fr/Lab06_FR.ipynb) | [Neural networks, gradients and learning](ML_Labs_2026/notebooks/en/Lab06_EN.ipynb) |
| 07 | [Prochain token et attention causale](ML_Labs_2026/notebooks/fr/Lab07_FR.ipynb) | [Next token and causal attention](ML_Labs_2026/notebooks/en/Lab07_EN.ipynb) |
| 08 | [Synthèse et évaluation finale](ML_Labs_2026/notebooks/fr/Lab08_FR.ipynb) | [Synthesis and final evaluation](ML_Labs_2026/notebooks/en/Lab08_EN.ipynb) |

Les niveaux A et B constituent le travail principal. Le niveau C propose une exploration facultative. Expliquer les entrées, les axes, les unités et le sens du résultat accompagne chaque exécution.
