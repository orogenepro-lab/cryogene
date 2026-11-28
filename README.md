# CRYOGÈNE — le froid au creux des vallées, prévu jusqu'à 7 jours

Frère hivernal d'OROGÈNE : prévision à haute résolution des **trous à froid**, des **inversions de température**,
des Tmin et des précipitations sur le Puy-de-Dôme, le Cantal et la Corrèze, de J+0 à J+7.

En hiver, par temps calme et ciel dégagé, l'air froid, plus lourd, descend la nuit le long des pentes et s'accumule
au fond des vallées et des cuvettes : on peut y relever 5 à 10 °C de moins que sur les hauteurs voisines. C'est
l'inversion de température. Elle explique les gelées dans les creux, la grisaille tenace sur la Limagne alors que le
soleil brille sur les puys, et l'accumulation de la pollution dans les villes de vallée. Les modèles météo habituels
voient le relief à plusieurs kilomètres près et passent à côté de ces contrastes.

## Principe

1. **Prévision ECMWF IFS** (open data, runs 00/12 UTC, jusqu'à +180 h).
2. **Profil vertical** au-dessus de chaque point : force et hauteur de l'inversion, stabilité, vent, nébulosité
   (`profile_inv.py`), et température du profil ramenée à l'altitude réelle du point (*T_base*).
3. **Correction apprise** (`train.py`, scikit-learn `HistGradientBoostingRegressor`) : écart observé − T_base,
   appris sur 6 hivers (2020-2026) de réanalyse **ERA5** × **80 stations Météo-France**, avec 13 descripteurs
   du relief (TPI 500 m à 20 km, sky view factor, drainage d'air froid…, `terrain_features.py`).
4. **Probabilité d'inversion** par secteur (`train_inversion.py`) : probabilité que la station de vallée soit plus
   froide que la station d'altitude (Limagne, bassin d'Aurillac, Val d'Allier), régime ciel clair / stratus selon
   la nébulosité basse.
5. **Page interactive** (`forecast_html.py`) : carte à 300 m (relief 100 m) toutes les 6 h, clic par station,
   courbes, isohyètes, frise des inversions.

Précision mesurée (validation croisée) : ~1,2 °C aux stations (1,6 °C les matins d'inversion), ~1,5 °C entre
les stations (2,6 °C par inversion), à quoi s'ajoute l'erreur de la prévision IFS qui croît avec l'échéance.
**Outil expérimental** : il ne remplace pas les prévisions et vigilances de Météo-France.

## Publication automatique (code privé, page publique)

Le code, les modèles et le relief restent dans ce dépôt **privé**. Le workflow `.github/workflows/cryogene.yml`
tourne 4 fois par jour et à la demande (onglet *Actions* → *CRYOGENE - prevision quotidienne* → *Run workflow*) :
il télécharge le run IFS, calcule la prévision et pousse **uniquement la page HTML** vers le dépôt public
`cryogene-site`, servi par GitHub Pages (un seul commit, écrasé à chaque run).

Mise en place (une fois) :
1. Sur le PC : `python export_for_github.py --res 300`, puis copier `cryogene_data\` dans `data/` de ce dépôt.
2. Créer le dépôt public vide `cryogene-site` (*Settings* → *Pages* → *Deploy from a branch* → `main` / `/ (root)`).
3. Créer un jeton fine-grained limité à `cryogene-site` (*Contents : Read and write*) et l'ajouter ici comme
   secret `SITE_TOKEN` (*Settings* → *Secrets and variables* → *Actions*).
4. Lancer le workflow une première fois à la main.

## Chaîne complète (sur le PC)

| Étape | Script | Fréquence |
|---|---|---|
| Relief (TPI, SVF, drainage) | `terrain_features.py` | une fois |
| Observations Météo-France | `obs_mf.py` | chaque semaine (vérification) |
| Réanalyse ERA5 (Earth Data Hub) | `era5_edh.py` | une fois par an |
| Table d'apprentissage | `build_dataset.py` | une fois par an |
| Modèle de température | `train.py` | une fois par an |
| Modèle d'inversion | `train_inversion.py` | une fois par an |
| Prévision + page | `forecast_html.py` (ou `cuvette_jour.bat`) | 2 fois par jour, ou via GitHub |

Après un réentraînement, relancer `export_for_github.py` et pousser `data/`.

## Données

Prévisions ECMWF open data (CC BY 4.0) · réanalyse ERA5, Copernicus Climate Change Service, via Destination Earth
Earth Data Hub · observations horaires Météo-France (data.gouv.fr, Licence Ouverte) · relief et limites
administratives IGN (Géoplateforme).
