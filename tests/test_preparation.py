"""Tests du nettoyage des données Give Me Some Credit.

Ce module porte toute la logique métier du projet : une régression silencieuse
ici fausserait les trois notebooks sans qu'aucune erreur ne soit levée. Les cas
couverts sont ceux établis par l'exploration (`notebooks/01_eda.ipynb`).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.preparation import (
    COMPTEURS_INCIDENTS,
    UTILISATION_MAX_PLAUSIBLE,
    NettoyageCredit,
)


@pytest.fixture
def dossiers():
    """Six dossiers couvrant chaque cas limite, un par ligne.

    0 : dossier normal
    1 : codes de saisie 96/98 dans les compteurs
    2 : revenu manquant, donc DebtRatio non calculable
    3 : revenu nul, même conséquence
    4 : âge impossible et utilisation du crédit aberrante
    5 : utilisation supérieure à 1 mais plausible, à ne pas écrêter
    """
    return pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [0.35, 0.50, 0.20, 0.80, 50708.0, 1.40],
        "age": [45, 52, 38, 61, 0, 29],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 98, 1, 0, 2, 0],
        "DebtRatio": [0.37, 0.42, 1250.0, 980.0, 0.55, 0.61],
        "MonthlyIncome": [5400.0, 6200.0, np.nan, 0.0, 3100.0, 4800.0],
        "NumberOfOpenCreditLinesAndLoans": [8, 5, 12, 3, 7, 9],
        "NumberOfTimes90DaysLate": [0, 98, 0, 0, 0, 1],
        "NumberRealEstateLoansOrLines": [1, 0, 2, 1, 0, 1],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 96, 0, 0, 1, 0],
        "NumberOfDependents": [2.0, 0.0, 3.0, 1.0, np.nan, 0.0],
    })


@pytest.fixture
def nettoye(dossiers):
    return NettoyageCredit().fit_transform(dossiers)


class TestCodesDeSaisie:
    """Les valeurs 96 et 98 sont des codes, pas des retards de paiement."""

    def test_les_codes_deviennent_manquants(self, nettoye):
        assert nettoye.loc[1, COMPTEURS_INCIDENTS].isna().all()

    def test_les_vrais_compteurs_sont_preserves(self, nettoye, dossiers):
        # La ligne 4 a 2 retards à 30-59 jours et 1 à 60-89 : des valeurs réelles.
        assert nettoye.loc[4, "NumberOfTime30-59DaysPastDueNotWorse"] == 2
        assert nettoye.loc[5, "NumberOfTimes90DaysLate"] == 1

    def test_l_indicatrice_marque_exactement_les_lignes_concernees(self, nettoye):
        assert nettoye["code_saisie"].tolist() == [0, 1, 0, 0, 0, 0]

    def test_l_information_n_est_pas_perdue(self, nettoye):
        """Neutraliser la valeur sans conserver le fait serait une perte :
        ces dossiers font 54,6 % de défaut dans les données réelles."""
        assert nettoye.loc[1, "code_saisie"] == 1
        assert nettoye.loc[1, COMPTEURS_INCIDENTS].isna().all()


class TestDebtRatio:
    """Le champ contient un montant brut quand le revenu est inexploitable."""

    def test_revenu_manquant_invalide_le_ratio(self, nettoye):
        assert np.isnan(nettoye.loc[2, "DebtRatio"])
        assert nettoye.loc[2, "ratio_non_calculable"] == 1

    def test_revenu_nul_invalide_aussi_le_ratio(self, nettoye):
        assert np.isnan(nettoye.loc[3, "DebtRatio"])
        assert nettoye.loc[3, "ratio_non_calculable"] == 1

    def test_les_vrais_ratios_sont_intacts(self, nettoye, dossiers):
        conserves = [0, 1, 4, 5]
        pd.testing.assert_series_equal(
            nettoye.loc[conserves, "DebtRatio"],
            dossiers.loc[conserves, "DebtRatio"],
        )
        assert nettoye.loc[conserves, "ratio_non_calculable"].eq(0).all()


class TestAberrations:
    def test_l_age_impossible_devient_manquant(self, nettoye):
        assert np.isnan(nettoye.loc[4, "age"])

    def test_les_ages_valides_sont_intacts(self, nettoye, dossiers):
        valides = [0, 1, 2, 3, 5]
        assert nettoye.loc[valides, "age"].tolist() == dossiers.loc[valides, "age"].tolist()

    def test_l_utilisation_aberrante_est_ecretee_et_signalee(self, nettoye):
        assert nettoye.loc[4, "RevolvingUtilizationOfUnsecuredLines"] == UTILISATION_MAX_PLAUSIBLE
        assert nettoye.loc[4, "utilisation_aberrante"] == 1

    def test_une_utilisation_superieure_a_un_reste_intacte(self, nettoye):
        """Dépasser 1 est plausible (dépassement de plafond, intérêts
        capitalisés) et reste très prédictif : on n'écrête qu'au-delà de 2."""
        assert nettoye.loc[5, "RevolvingUtilizationOfUnsecuredLines"] == 1.40
        assert nettoye.loc[5, "utilisation_aberrante"] == 0

    def test_aucune_ligne_n_est_supprimee(self, nettoye, dossiers):
        assert len(nettoye) == len(dossiers)


class TestAbsenceDeFuite:
    """Le transformateur ne doit rien apprendre des données qu'il voit."""

    def test_le_resultat_ne_depend_pas_du_jeu_d_ajustement(self, dossiers):
        sur_tout = NettoyageCredit().fit(dossiers).transform(dossiers)
        # Ajusté sur deux lignes anodines, appliqué au jeu complet.
        sur_extrait = NettoyageCredit().fit(dossiers.iloc[:2]).transform(dossiers)
        pd.testing.assert_frame_equal(sur_tout, sur_extrait)

    def test_le_seuil_d_ecretage_est_un_parametre_pas_une_statistique(self, dossiers):
        """Un plafond appris sur un quantile dépendrait de l'échantillon et
        rouvrirait la porte à une fuite. Il doit rester explicite."""
        assert NettoyageCredit().get_params()["plafond_utilisation"] == UTILISATION_MAX_PLAUSIBLE

    def test_les_donnees_d_entree_ne_sont_pas_modifiees(self, dossiers):
        avant = dossiers.copy(deep=True)
        NettoyageCredit().fit_transform(dossiers)
        pd.testing.assert_frame_equal(dossiers, avant)


class TestContratScikitLearn:
    def test_les_colonnes_annoncees_correspondent_a_la_sortie(self, dossiers):
        transformateur = NettoyageCredit().fit(dossiers)
        attendues = list(transformateur.get_feature_names_out())
        assert attendues == list(transformateur.transform(dossiers).columns)

    def test_trois_indicatrices_sont_ajoutees(self, dossiers, nettoye):
        nouvelles = set(nettoye.columns) - set(dossiers.columns)
        assert nouvelles == {"code_saisie", "ratio_non_calculable", "utilisation_aberrante"}

    def test_utilisable_dans_un_pipeline(self, dossiers):
        chaine = Pipeline([("nettoyage", NettoyageCredit())])
        assert chaine.fit_transform(dossiers).shape == (6, 13)

    def test_clonable(self, dossiers):
        """clone() est ce qu'appelle la validation croisée à chaque pli."""
        copie = clone(NettoyageCredit(plafond_utilisation=3.0))
        assert copie.plafond_utilisation == 3.0

    def test_un_plafond_personnalise_est_respecte(self, dossiers):
        resultat = NettoyageCredit(plafond_utilisation=1.0).fit_transform(dossiers)
        assert resultat["RevolvingUtilizationOfUnsecuredLines"].max() == 1.0
        assert resultat["utilisation_aberrante"].tolist() == [0, 0, 0, 0, 1, 1]


class TestEntreesInvalides:
    def test_refuse_autre_chose_qu_un_dataframe(self, dossiers):
        with pytest.raises(TypeError, match="DataFrame"):
            NettoyageCredit().fit(dossiers.to_numpy())

    def test_signale_les_colonnes_absentes(self, dossiers):
        with pytest.raises(ValueError, match="DebtRatio"):
            NettoyageCredit().fit(dossiers.drop(columns="DebtRatio"))
