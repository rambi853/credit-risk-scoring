"""Genere la soumission Kaggle a partir du modele retenu.

Le modele est reentraine sur les 150 000 dossiers etiquetes : le decoupage
70/30 des notebooks servait a estimer la performance, plus a la produire.

Envoi (le jeton au format KGAT exige KAGGLE_API_TOKEN, le client ne sait pas
l'exploiter depuis ~/.kaggle/kaggle.json) :

    KAGGLE_API_TOKEN=... kaggle competitions submit \\
        -c GiveMeSomeCredit -f submission.csv -m "..."
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from src.preparation import NettoyageCredit, charger_donnees

BRUT = str(Path(__file__).resolve().parents[1] / "data" / "raw") + "/"
X, y = charger_donnees(BRUT + "cs-training.csv")

test = pd.read_csv(BRUT + "cs-test.csv", index_col=0)
ids = test.index
test = test.drop(columns=["SeriousDlqin2yrs"])

chaine = Pipeline([
    ("nettoyage", NettoyageCredit()),
    ("modele", LGBMClassifier(subsample=1.0, reg_lambda=10, num_leaves=15,
                              n_estimators=300, min_child_samples=100,
                              learning_rate=0.05, colsample_bytree=1.0,
                              subsample_freq=1, random_state=0, verbose=-1, n_jobs=4)),
])
# Entraînement sur les 150 000 dossiers : plus rien à garder de côté,
# l'évaluation se fera sur le jeu de test de la compétition.
chaine.fit(X, y)
p = chaine.predict_proba(test)[:, 1]

sortie = pd.DataFrame({"Id": ids, "Probability": p})
chemin = str(Path(__file__).resolve().parents[1] / "submission.csv")
sortie.to_csv(chemin, index=False)
print("lignes         :", len(sortie))
print("probabilités   : min %.5f | médiane %.5f | max %.5f" % (p.min(), pd.Series(p).median(), p.max()))
print("taux moyen     : %.4f (attendu ~0,067)" % p.mean())
