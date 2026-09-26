# =============================================================================
# Couche d'accès aux données (I/O uniquement, pas de logique métier ici).
# =============================================================================
# Toutes les données consommées par l'app sont déjà nettoyées/pré-calculées
# en amont (notebook d'exploration) et stockées dans data/processed/ :
# ces fonctions se contentent de les charger depuis le disque. Elles sont
# appelées une seule fois par app/app.py, dans load_all_data() (mise en
# cache Streamlit), donc pas besoin de mettre son propre cache ici.
import pandas as pd
from pathlib import Path

# Dossiers (chemins calculés depuis l'emplacement de ce fichier, app/utils.py,
# pour que l'app fonctionne quel que soit le répertoire depuis lequel on lance
# `streamlit run app/app.py`)
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_df_clean():
    """Charge le dataset de transactions nettoyées (1 ligne = 1 ligne de
    facture), au format parquet pour un chargement rapide malgré le volume
    (~1 million de lignes)."""
    path = PROCESSED_DIR / "online_retail_clean.parquet"
    df = pd.read_parquet(path)
    return df


def load_rfm():
    """Charge la table RFM : une ligne par client, avec ses scores
    Recency/Frequency/Monetary déjà calculés et son segment déjà assigné."""
    path = PROCESSED_DIR / "rfm_segments.csv"
    df = pd.read_csv(path, index_col=0)
    return df


def load_cohort_retention():
    """Charge la matrice de rétention par cohorte : une ligne par cohorte
    (mois de 1ère commande), une colonne par âge en mois, valeur = %
    de clients encore actifs à cet âge."""
    path = PROCESSED_DIR / "cohort_retention.csv"
    df = pd.read_csv(path, index_col=0)
    return df


def load_cohort_avg_revenue():
    """Charge le revenu moyen généré par âge de cohorte (même structure que
    load_cohort_retention, mais avec un montant de revenu au lieu d'un %)."""
    path = PROCESSED_DIR / "cohort_avg_revenue.csv"
    df = pd.read_csv(path, index_col=0)
    return df
