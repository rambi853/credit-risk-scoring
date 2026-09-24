# CLAUDE.md

## Le projet

Compétition Kaggle **Give Me Some Credit** : prédire la probabilité qu'un
emprunteur connaisse un incident de paiement grave dans les deux ans
(classification binaire, classes déséquilibrées ~7 % de positifs).

Deux rendus : un **notebook** d'analyse et des **slides**.

## Objectif, et ce qu'il implique

Ce projet est une remise en jambes avant une reprise de poste en data science.
**Le score compte moins que la démarche.**

- Expliquer le raisonnement à chaque étape plutôt que livrer un bloc de code fini.
- Couvrir large : EDA, feature engineering, validation, interprétabilité.
- Les slides s'adressent à un **public non technique** : problème, impact chiffré,
  recommandations. Pas de détails de modélisation.

## Commandes

```bash
source .venv/bin/activate         # Python 3.9 — ou ../.venv si le venv est partagé avec d'autres projets
jupyter lab

# Récupérer les données (nécessite ~/.kaggle/kaggle.json, chmod 600)
kaggle competitions download -c GiveMeSomeCredit -p data/raw

pytest                            # tests de src/preparation.py
```

## Structure

```
data/raw/          données Kaggle brutes — jamais versionnées
data/processed/    jeux nettoyés — jamais versionnés
notebooks/         analyse, numérotés par ordre d'exécution (01_eda.ipynb...)
src/               fonctions réutilisables, importées par les notebooks
reports/figures/   graphiques exportés, source des visuels des slides
slides/            présentation finale
```

## Conventions

- Toute fonction réutilisée par plus d'un notebook part dans `src/`, et ce qui
  part dans `src/` est testé : ce module porte la logique métier, une
  régression y serait silencieuse.
- Le test `test_le_resultat_ne_depend_pas_du_jeu_d_ajustement` garde la porte
  fermée à la fuite de données : rien dans le nettoyage ne doit être appris
  sur les données transformées.
- Les graphiques destinés aux slides sont exportés dans `reports/figures/`,
  pas seulement affichés en sortie de cellule.
- Métrique de référence : **AUC**, jamais l'accuracy (les classes sont
  déséquilibrées, un modèle constant atteindrait 93 %).
- Le choix du seuil de décision est un arbitrage métier : il se justifie en
  coût (défauts évités vs bons clients refusés), pas en F1.

## Contraintes de la machine

macOS 12.7.4, Intel sans GPU, ~10 Go de disque libre.

**Rien ne compile sur cette machine** : Homebrew ne fournit plus de binaires
précompilés pour macOS 12, et les Command Line Tools sont cassés (ni `clang`
ni `xcrun`). Toujours privilégier les paquets conda-forge, les wheels PyPI ou
les binaires officiels — un `brew install` échouera après une longue attente.

Pas de deep learning lourd : s'en tenir à scikit-learn et LightGBM.

L'environnement est un **venv + pip**, pas conda : le solveur de conda 4.10 sur
conda-forge tourne plus de 17 minutes sans aboutir sur cette machine. Python 3.9
est en fin de support, mais les versions de pandas et scikit-learn installées
sont récentes, ce qui est ce qui compte pour les APIs.

Client Kaggle : le token est au nouveau format (`KGAT_`), qui s'envoie en
**Bearer** et non en Basic. Le client `kaggle` historique ne sait pas le faire ;
télécharger via `curl -H "Authorization: Bearer $KAGGLE_KEY"`.
