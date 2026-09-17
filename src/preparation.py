"""Nettoyage des données Give Me Some Credit.

Les décisions appliquées ici viennent de `notebooks/01_eda.ipynb`, qui établit
pourquoi chacune est nécessaire. Le transformateur respecte l'API scikit-learn
pour être utilisable dans un Pipeline : tout ce qui doit être appris (le seuil
d'écrêtage) l'est dans `fit`, donc sur le seul jeu d'entraînement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COMPTEURS_INCIDENTS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

CODE_SAISIE_MIN = 90
AGE_MINIMUM = 18

# Au-dela de 2, le taux d'utilisation du credit n'est plus interpretable : le
# risque de defaut, qui monte regulierement jusqu'a 44,5 % entre 1,5 et 2,
# retombe a 7,1 % au-dela de 10, soit le taux moyen de la population. Ces
# valeurs ne decrivent donc pas des clients surendettes mais des saisies
# corrompues.
UTILISATION_MAX_PLAUSIBLE = 2.0


class NettoyageCredit(BaseEstimator, TransformerMixin):
    """Corrige les trois défauts de qualité identifiés à l'exploration.

    1. Les valeurs 96 et 98 des compteurs d'incidents sont des codes de saisie,
       pas des retards. On les passe en NaN — sinon le modèle lit « 98 retards »
       — tout en conservant l'information dans une indicatrice, car ces
       dossiers font 54,6 % de défaut.
    2. `DebtRatio` contient un montant brut au lieu d'un ratio lorsque le revenu
       est absent ou nul. On isole ces lignes plutôt que de laisser le modèle
       comparer des dollars à des ratios.
    3. Les valeurs aberrantes (âge à 0, taux d'utilisation jusqu'à 50 708) sont
       écrêtées et signalées, sans perdre de lignes.

    Parameters
    ----------
    plafond_utilisation :
        Valeur au-delà de laquelle le taux d'utilisation du crédit est écrêté et
        signalé comme aberrant. Voir `UTILISATION_MAX_PLAUSIBLE`.
    """

    def __init__(self, plafond_utilisation: float = UTILISATION_MAX_PLAUSIBLE):
        self.plafond_utilisation = plafond_utilisation

    def fit(self, X: pd.DataFrame, y=None):
        X = self._verifier(X)
        self.colonnes_entree_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._verifier(X).copy()

        # 1. Codes de saisie : indicatrice d'abord, neutralisation ensuite.
        code = (X[COMPTEURS_INCIDENTS] >= CODE_SAISIE_MIN).any(axis=1)
        X["code_saisie"] = code.astype("int8")
        X[COMPTEURS_INCIDENTS] = X[COMPTEURS_INCIDENTS].mask(
            X[COMPTEURS_INCIDENTS] >= CODE_SAISIE_MIN
        )

        # 2. DebtRatio : le champ n'est un ratio que si le revenu est exploitable.
        revenu_inexploitable = X["MonthlyIncome"].isna() | (X["MonthlyIncome"] == 0)
        X["ratio_non_calculable"] = revenu_inexploitable.astype("int8")
        X.loc[revenu_inexploitable, "DebtRatio"] = np.nan

        # 3. Aberrations.
        X.loc[X["age"] < AGE_MINIMUM, "age"] = np.nan

        # On écrête plutôt que de supprimer, et on signale : au-delà du plafond
        # ces dossiers reviennent au risque moyen, ils forment un groupe à part.
        utilisation = X["RevolvingUtilizationOfUnsecuredLines"]
        X["utilisation_aberrante"] = (utilisation > self.plafond_utilisation).astype("int8")
        X["RevolvingUtilizationOfUnsecuredLines"] = utilisation.clip(
            upper=self.plafond_utilisation
        )

        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            list(self.colonnes_entree_)
            + ["code_saisie", "ratio_non_calculable", "utilisation_aberrante"],
            dtype=object,
        )

    @staticmethod
    def _verifier(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "NettoyageCredit attend un DataFrame : il travaille par nom de colonne."
            )
        manquantes = set(COMPTEURS_INCIDENTS + ["DebtRatio", "MonthlyIncome", "age"]) - set(X.columns)
        if manquantes:
            raise ValueError(f"Colonnes absentes du DataFrame : {sorted(manquantes)}")
        return X


def charger_donnees(chemin) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le fichier d'entraînement et sépare la cible des variables."""
    df = pd.read_csv(chemin, index_col=0)
    y = df.pop("SeriousDlqin2yrs")
    return df, y
