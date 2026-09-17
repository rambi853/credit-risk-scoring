# Give Me Some Credit — prédiction de défaut de crédit

Projet Kaggle tabulaire : prédire la probabilité qu'un emprunteur connaisse un
incident de paiement grave dans les deux ans.

**Rendus attendus :** un notebook d'analyse et un jeu de slides à destination
d'un public non technique.

## Structure

```
data/raw/          données brutes Kaggle (non versionnées)
data/processed/    jeux de données nettoyés (non versionnés)
notebooks/         analyse exploratoire et modélisation
src/               fonctions réutilisables importées par les notebooks
reports/figures/   graphiques exportés pour les slides
slides/            présentation finale
```

## Environnement

```bash
source .venv/bin/activate
pip install -r requirements.txt   # première installation
jupyter lab
```

## Données

Le dataset n'est pas versionné. Pour le récupérer :

```bash
kaggle competitions download -c GiveMeSomeCredit -p data/raw
unzip data/raw/GiveMeSomeCredit.zip -d data/raw
```

Nécessite un token API Kaggle dans `~/.kaggle/kaggle.json`
(kaggle.com → Settings → API → Create New Token).
