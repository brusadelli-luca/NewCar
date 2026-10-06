# New Car Project

Projet de régression linéaire en Python (notebook Jupyter) : prédire le prix de vente d'un véhicule d'occasion.

Code écrit de décembre 2022 à janvier 2023. Revue et corrections faites avec Claude en octobre 2026 (voir les commits marqués `Co-Authored-By: Claude`).

## Prérequis
* Python 3 (testé avec Python 3.12 et 3.14)
* pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, missingno et jupyter, listés dans `requirements.txt`

Depuis le dossier du projet, créer un environnement virtuel et installer les bibliothèques (commandes Windows) :

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Lancement
```
.venv\Scripts\python -m jupyter notebook main.ipynb
```
puis exécuter toutes les cellules (menu *Run > Run All Cells*). Les sorties enregistrées dans le notebook sont celles de la dernière exécution.

## Données
`data/carData.csv` : jeu de données « Vehicle dataset » de CarDekho (301 véhicules). Colonnes utilisées :
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
5. Estimation du prix d'un véhicule de 7 ans, 100 000 km, boîte manuelle.

## Résultats (validation)
| Modèle | R² apprentissage | R² validation | RMSE (lakh) |
|---|---|---|---|
| Simple (scipy, numpy, sklearn, statsmodels) | 0,070 | 0,002 | 5,61 |
| Personnalisé | 0,070 | 0,002 | 5,61 |
| Multivarié | 0,283 | 0,280 | 4,77 |

Les quatre modèles simples donnent la même droite (pente -0,370, ordonnée 6,151) ; le modèle personnalisé la retrouve. Le modèle multivarié est le meilleur ; il estime le véhicule de la question 5 à environ 6,87 lakh.

## Limites
* L'âge seul explique très peu le prix (R² proche de 0) : le prix dépend surtout du modèle de voiture, non utilisé ici.
* La première estimation de la question 5 utilise des coefficients recopiés à la main d'un calcul précédent ; seule la seconde est recalculée.
