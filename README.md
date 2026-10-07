# New Car Project

Projet de régression linéaire en Python (notebook Jupyter) : prédire le prix de vente d'un véhicule d'occasion.

Code écrit de décembre 2022 à janvier 2023. Revue et corrections faites avec Claude en octobre 2026 (voir les commits marqués `Co-Authored-By: Claude`).

## Prérequis
* Python 3 (testé avec Python 3.12 et 3.14)
* pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, missingno, listés dans `requirements.txt`

Depuis le dossier du projet, créer un environnement virtuel et installer les bibliothèques (commandes Windows) :

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Lancement
Ouvrir `main.ipynb` avec VS Code (extension Jupyter, en choisissant le Python du `.venv`), ou avec JupyterLab :

```
.venv\Scripts\python -m pip install jupyterlab
.venv\Scripts\python -m jupyter lab main.ipynb
```

puis exécuter toutes les cellules (« Run All »). Les sorties (tableaux, graphiques, résultats) sont enregistrées dans le notebook : GitHub les affiche sans rien exécuter. Elles correspondent à une exécution complète avec Python 3.12.

## Données
`data/carData.csv` : jeu de données [« Vehicle dataset from cardekho »](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) publié sur Kaggle par nehalbirla, sous licence Database Contents License (DbCL) v1.0 (301 véhicules). Colonnes utilisées :
* `Year` : année de construction (l'âge est calculé comme `année max + 1 - Year`)
* `Selling_Price` : prix de vente en lakh (100 000 roupies), la valeur à prédire
* `Kms_Driven` : kilomètres parcourus
* `Transmission` : `Manual` ou `Automatic` (codé 1 et 0)
* `Owner` : nombre de propriétaires précédents

Les véhicules de plus de 300 000 km, vendus plus de 25 lakh ou ayant eu 3 propriétaires sont retirés avant l'apprentissage (297 véhicules restants), puis le jeu est découpé en 80 % apprentissage et 20 % validation (`random_state = 42`).

## Contenu du notebook
1. Exploration des données (valeurs manquantes, histogrammes, matrice de corrélation).
2. Régression linéaire simple, prix en fonction de l'âge, avec 4 bibliothèques : scipy (`linregress`), numpy (`polyfit`), scikit-learn (`LinearRegression`) et statsmodels (`OLS`).
3. Régression multivariée (scikit-learn) avec l'âge, les kilomètres et la transmission.
4. Modèle personnalisé (`CustomLinearRegression`) : descente de gradient, arrêtée quand l'erreur baisse de moins de 0,000001.
5. Prix d'un véhicule de moins de 7 ans, moins de 100 000 km, boîte manuelle : prédiction pour 7 ans et 100 000 km (avec tout le jeu de données, puis sans les données isolées), et prix moyen des véhicules du jeu de données qui correspondent.

## Résultats (validation)
| Modèle | R² apprentissage | R² validation | RMSE (lakh) |
|---|---|---|---|
| Simple (scipy, numpy, sklearn, statsmodels) | 0,070 | 0,002 | 5,61 |
| Personnalisé | 0,070 | 0,002 | 5,61 |
| Multivarié | 0,283 | 0,280 | 4,77 |

Les quatre modèles simples donnent la même droite (pente -0,370, ordonnée 6,151) ; le modèle personnalisé la retrouve. Le modèle multivarié est le meilleur.

Question 5 : le modèle multivarié prédit 6,87 lakh pour un véhicule d'exactement 7 ans et 100 000 km (3,98 lakh avec tout le jeu de données). Les 189 véhicules du jeu de données de moins de 7 ans, moins de 100 000 km et boîte manuelle valent en moyenne 4,34 lakh (4,34 prédit) : c'est la meilleure réponse à la question posée.

## Limites
* L'âge seul explique très peu le prix (R² proche de 0). Le prix du véhicule neuf (`Present_Price`, corrélation 0,89 avec le prix de vente) n'est pas utilisé : l'ajouter au modèle multivarié porterait le R² de validation à 0,88 (vérifié hors du notebook).
* Dans le modèle multivarié, le coefficient des kilomètres est positif (+6,6 lakh pour 100 000 km) : les voitures chères roulent plus, et le modèle attribue aux kilomètres un effet qui vient en fait du prix neuf. C'est pourquoi la prédiction à 100 000 km est élevée.
