import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from utils import (
    load_df_clean,
    load_rfm,
    load_cohort_retention,
    load_cohort_avg_revenue,
)

# ---------- CONFIG GÉNÉRALE ----------
st.set_page_config(
    page_title="Online Retail II – Dashboard marketing",
    page_icon="📊",
    layout="wide",
)

# ---- PALETTE & STYLE VISUEL (charte cohérente clair/sombre) ----
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}

# Rampe séquentielle mono-teinte (bleu, clair -> foncé), utilisée pour les
# heatmaps et nuages de points afin de garder une seule échelle de couleur
# cohérente sur tout le dashboard plutôt que des colormaps génériques.
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["ink_secondary"],
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "axes.facecolor": PALETTE["surface"],
        "figure.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
    }
)


def style_axes(ax):
    """Applique le style du dashboard (spines fines, grille discrète) à un axe matplotlib."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)


# ---- CSS : cartes KPI + expanders, cohérents clair/sombre ----
st.markdown(
    """
    <style>
    :root {
        --accent: #2a78d6;
        --surface: #fcfcfb;
        --ink: #0b0b0b;
        --ink-secondary: #52514e;
        --border: rgba(11,11,11,0.10);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #3987e5;
            --surface: #1a1a19;
            --ink: #ffffff;
            --ink-secondary: #c3c2b7;
            --border: rgba(255,255,255,0.10);
        }
    }

    /* Cartes pour les st.metric */
    div[data-testid="stMetric"] {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--ink-secondary) !important;
    }

    /* Header de l'expander (fermé) */
    .streamlit-expanderHeader {
        background-color: var(--surface) !important;
        color: var(--accent) !important;
        border-radius: 8px;
        border: 1px solid var(--border) !important;
        font-weight: 600;
        padding: 8px 12px;
    }

    /* Header au survol */
    .streamlit-expanderHeader:hover {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        cursor: pointer;
    }

    /* Contenu de l'expander (ouvert) */
    .streamlit-expanderContent {
        background-color: var(--surface) !important;
        color: var(--ink) !important;
        border-left: 1px solid var(--border) !important;
        border-right: 1px solid var(--border) !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 0 8px 8px;
        padding: 12px 14px;
    }

    hr {
        margin: 1.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Online Retail II – Dashboard marketing")

st.markdown(
    """
Cette application répond au besoin de l’équipe marketing de *Online Retail* :

- suivre la *rétention par cohortes* (qualité de l’onboarding et de la fidélisation),
- analyser les *segments clients (RFM)* pour prioriser les actions CRM,
- estimer la *Customer Lifetime Value (CLV)* avec une approche empirique et une *formule fermée*,
- tester des *scénarios business* (impact d’une amélioration de la rétention, de la marge…),
- générer une *liste de clients activables* pour les campagnes.
"""
)

# ---------- CHARGEMENT DES DONNÉES ----------
@st.cache_data
def load_all_data():
    df_clean = load_df_clean()
    rfm = load_rfm()
    cohort_retention = load_cohort_retention()
    cohort_avg_rev = load_cohort_avg_revenue()
    return df_clean, rfm, cohort_retention, cohort_avg_rev


df_clean, rfm, cohort_retention, cohort_avg_rev = load_all_data()

# ---------- FONCTIONS UTILITAIRES ----------


def compute_amount_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne Amount si elle n'existe pas déjà."""
    df = df.copy()
    if "Amount" not in df.columns:
        df["Amount"] = df["Quantity"] * df["Price"]
    return df


def get_date_bounds(df: pd.DataFrame):
    """Trouve la date min et max pour la période d'analyse."""
    col_date = "InvoiceDate"
    min_date = df[col_date].min().date()
    max_date = df[col_date].max().date()
    return col_date, min_date, max_date


def filter_transactions(
    df: pd.DataFrame,
    col_date: str,
    start_date,
    end_date,
    countries,
    returns_mode: str = "Inclure",
    min_amount: float = 0.0,
):
    """Applique les filtres de base sur les transactions."""
    df = df.copy()

    # Filtre dates
    mask_date = (df[col_date] >= pd.to_datetime(start_date)) & (
        df[col_date] <= pd.to_datetime(end_date)
    )
    df = df[mask_date]

    # Filtre pays
    if countries:
        df = df[df["Country"].isin(countries)]

    # Gestion des retours (factures commençant par 'C')
    if "Invoice" in df.columns:
        invoice_col = "Invoice"
    else:
        invoice_col = "InvoiceNo"  # au cas où

    is_return = df[invoice_col].astype(str).str.startswith("C")

    if returns_mode == "Exclure":
        df = df[~is_return]
    elif returns_mode == "Neutraliser":
        df.loc[is_return, "Quantity"] = df.loc[is_return, "Quantity"].abs()

    df = compute_amount_if_needed(df)

    # Seuil minimum de montant par transaction
    if min_amount > 0:
        df = df[df["Amount"] >= min_amount]

    return df


# ---------- PAGES ----------


def page_overview(
    df_clean: pd.DataFrame,
    df_filtered: pd.DataFrame,
    cohort_avg_rev: pd.DataFrame,
    rfm: pd.DataFrame,
):
    """Page 1 – KPIs globaux + premiers graphiques."""

    st.header("Overview – KPIs globaux")

    st.markdown(
        """
Cette page donne une *vue d’ensemble* de la performance du portefeuille clients :

- combien de *clients actifs* sur la période sélectionnée,
- quel *revenu net* ils génèrent,
- une estimation de la *CLV moyenne* à partir des cohortes,
- une *North Star Metric* : le revenu moyen généré au *3ᵉ mois* après la première commande,
- un focus sur la *segmentation RFM* et l’évolution du *CA par âge de cohorte*.

Les filtres à gauche permettent de changer la période, les pays ou la gestion des retours.
"""
    )

    # --- Sécuriser les montants ---
    df_filtered = compute_amount_if_needed(df_filtered)

    # --- KPIs de base (sur données filtrées) ---
    nb_clients_actifs = df_filtered["Customer ID"].nunique()
    nb_invoices = df_filtered["Invoice"].nunique()
    ca_total = df_filtered["Amount"].sum()

    # CLV empirique baseline = somme du revenu moyen par âge de cohorte
    avg_revenue_per_age = cohort_avg_rev.mean(axis=0)
    clv_empirique = avg_revenue_per_age.sum()

    # Taille des segments RFM
    nb_segments_rfm = rfm["Segment"].nunique()
    nb_clients_segmentes = rfm.index.nunique()

    # North Star Metric : revenu moyen généré à M+3 par cohorte
    if "3" in cohort_avg_rev.columns:
        m3_col = "3"
    else:
        m3_col = cohort_avg_rev.columns[3]
    revenu_m3 = cohort_avg_rev[m3_col].mean(skipna=True)

    # --- Affichage des KPIs (ligne 1) ---
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Clients actifs (filtrés)", f"{nb_clients_actifs:,}")
        with st.expander("ℹ Définition"):
            st.write(
                "Nombre de clients ayant au moins *une transaction* "
                "dans la période et les filtres sélectionnés."
            )

    with c2:
        st.metric("Revenu total (net, filtré)", f"{ca_total:,.0f} £")
        with st.expander("ℹ Définition"):
            st.write(
                "Somme de Quantity × Price sur les transactions filtrées. "
                "Les retours (factures commençant par 'C') sont inclus "
                "en montant *négatif*."
            )

    with c3:
        st.metric("CLV baseline (empirique)", f"{clv_empirique:,.0f} £")
        with st.expander("ℹ Définition"):
            st.write(
                "CLV empirique = somme du *revenu moyen par âge de cohorte* "
                "(M0 → M12). C’est la valeur moyenne générée par un client "
                "sur toute sa durée de vie observée."
            )

    with c4:
        st.metric("North Star – Revenu moyen M+3", f"{revenu_m3:,.0f} £")
        with st.expander("ℹ Définition"):
            st.write(
                "Revenu moyen généré par les clients *au 3ᵉ mois* après "
                "leur première commande. C’est notre North Star Metric "
                "pour suivre la qualité de l’onboarding."
            )

    st.markdown("---")

    # --- Ligne 2 : Infos segments RFM + CA par âge de cohorte ---
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Segmentation RFM – synthèse")
        st.metric("Nombre de segments RFM", nb_segments_rfm)
        st.metric("Clients avec segment RFM", f"{nb_clients_segmentes:,}")
        with st.expander("ℹ Rappel RFM"):
            st.write(
                "- *Recency* : jours depuis la dernière commande\n"
                "- *Frequency* : nombre de commandes\n"
                "- *Monetary* : montant cumulé\n\n"
                "Les scores R/F/M sont regroupés en quantiles (1–5) puis "
                "mappés en segments (Champions, Loyaux, À risque…)."
            )

        # (petit exemple d’agrégat par segment si besoin)
        _ = (
            rfm.groupby("Segment")
            .size()
            .sort_values(ascending=False)
            .head(5)
        )

    with c6:
        st.subheader("CA moyen par âge de cohorte")
        avg_age = avg_revenue_per_age.reset_index()
        avg_age.columns = ["Age_mois", "Revenu_moyen"]
        try:
            avg_age["Age_mois"] = avg_age["Age_mois"].astype(int)
        except Exception:
            pass

        st.line_chart(
            avg_age.set_index("Age_mois")["Revenu_moyen"],
            height=250,
        )
        with st.expander("ℹ Interprétation"):
            st.write(
                "Cette courbe montre combien un client génère *en moyenne* "
                "à chaque âge de cohorte (M0, M1, M2, …). On peut repérer "
                "les âges où la valeur chute et prioriser les actions CRM."
            )

    st.markdown("---")

    # --- Aperçu des transactions filtrées ---
    st.subheader("Aperçu des transactions (100 premières lignes filtrées)")
    st.dataframe(df_filtered.head(100))

    # --- Export CSV des transactions filtrées ---
    st.subheader("Export des transactions filtrées")
    csv_filtered = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Télécharger les transactions filtrées (CSV)",
        data=csv_filtered,
        file_name="transactions_filtrees.csv",
        mime="text/csv",
    )


def page_cohortes(cohort_retention: pd.DataFrame, cohort_avg_rev: pd.DataFrame):
    """Page 2 – Cohortes (rétention + revenu)."""
    st.header("Cohortes – Rétention & revenu")

    st.markdown(
        """
Sur cette page, on analyse le comportement des *cohortes de clients* :

- la *rétention* : pourcentage de clients encore actifs à M+1, M+2, …  
- le *revenu moyen* généré à chaque âge de cohorte,  
- le *revenu cumulé* d’une cohorte au fil du temps.

L’objectif est d’identifier :
- les cohortes les plus rentables,
- les âges où les clients décrochent,
- les opportunités d’amélioration de l’onboarding et de la fidélisation.
"""
    )

    # ---------- 1. Heatmap de rétention ----------
    st.subheader("Heatmap de rétention (cohorte × âge)")

    retention_pct = cohort_retention.copy() * 100
    retention_pct.index.name = "Cohorte"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        retention_pct,
        ax=ax,
        cmap=BLUE_SEQUENTIAL,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8, "color": PALETTE["ink"]},
        linewidths=2,
        linecolor=PALETTE["surface"],
        cbar_kws={"label": "Rétention (%)"},
    )
    ax.set_xlabel("Âge de cohorte (mois)")
    ax.set_ylabel("Cohorte (mois de 1ʳᵉ commande)")
    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("ℹ Comment lire cette heatmap ?"):
        st.write(
            """
Chaque case correspond à *une cohorte* (ligne) et *un âge* (colonne) :

- Ligne = mois de première commande (cohorte)  
- Colonne = âge en mois après la première commande (M1, M2, …)  
- Valeur = % de clients de la cohorte encore actifs à cet âge.
"""
        )

    st.markdown("---")

    # ---------- 2. Courbe de rétention d'une cohorte ----------
    st.subheader("Courbe de rétention pour une cohorte")

    cohort_list = retention_pct.index.tolist()
    selected_cohort_ret = st.selectbox(
        "Choisir une cohorte pour voir sa courbe de rétention",
        cohort_list,
    )

    ret_curve = retention_pct.loc[selected_cohort_ret].dropna()
    try:
        ret_curve.index = ret_curve.index.astype(int)
    except Exception:
        pass

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(
        ret_curve.index,
        ret_curve.values,
        color=PALETTE["blue"],
        linewidth=2,
        marker="o",
        markersize=7,
        markerfacecolor=PALETTE["surface"],
        markeredgecolor=PALETTE["blue"],
        markeredgewidth=2,
    )
    ax2.set_xlabel("Âge de cohorte (mois)")
    ax2.set_ylabel("Rétention (%)")
    ax2.set_title(f"Courbe de rétention – Cohorte {selected_cohort_ret}", color=PALETTE["ink"], fontweight="bold")
    style_axes(ax2)
    fig2.tight_layout()
    st.pyplot(fig2)

    with st.expander("ℹ Interprétation de la courbe"):
        st.write(
            """
Cette courbe montre la *décroissance de la rétention* pour une cohorte donnée.
On voit à quel âge les clients décrochent le plus, ce qui permet de
prioriser les actions CRM (relance, promotions, programmes de fidélité…).
"""
        )

    st.markdown("---")

    # ---------- 3. Revenu moyen et cumulé d'une cohorte ----------
    st.subheader("Revenu moyen et cumulé par âge de cohorte")

    cohort_list_rev = cohort_avg_rev.index.tolist()
    selected_cohort_rev = st.selectbox(
        "Choisir une cohorte pour le revenu",
        cohort_list_rev,
    )

    rev = cohort_avg_rev.loc[selected_cohort_rev].dropna()
    try:
        rev.index = rev.index.astype(int)
    except Exception:
        pass

    rev_cum = rev.cumsum()

    col1, col2 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(
            rev.index,
            rev.values,
            color=PALETTE["blue"],
            linewidth=2,
            marker="o",
            markersize=7,
            markerfacecolor=PALETTE["surface"],
            markeredgecolor=PALETTE["blue"],
            markeredgewidth=2,
        )
        ax3.set_xlabel("Âge (mois)")
        ax3.set_ylabel("Revenu moyen (£)")
        ax3.set_title(f"Revenu moyen par âge – Cohorte {selected_cohort_rev}", color=PALETTE["ink"], fontweight="bold")
        style_axes(ax3)
        fig3.tight_layout()
        st.pyplot(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.plot(
            rev_cum.index,
            rev_cum.values,
            color=PALETTE["orange"],
            linewidth=2,
            marker="o",
            markersize=7,
            markerfacecolor=PALETTE["surface"],
            markeredgecolor=PALETTE["orange"],
            markeredgewidth=2,
        )
        ax4.set_xlabel("Âge (mois)")
        ax4.set_ylabel("Revenu cumulé (£)")
        ax4.set_title(f"Revenu cumulé – Cohorte {selected_cohort_rev}", color=PALETTE["ink"], fontweight="bold")
        style_axes(ax4)
        fig4.tight_layout()
        st.pyplot(fig4)

    with st.expander("ℹ Lecture du revenu cumulé"):
        st.write(
            """
Le *revenu cumulé* d’une cohorte permet de voir :

- combien elle a rapporté au total à M1, M3, M6, M12…  
- quelles cohortes sont les plus rentables sur la durée.

C’est directement lié à la *CLV* : plus la courbe est haute, plus la cohorte est précieuse.
"""
        )


def page_rfm(rfm: pd.DataFrame):
    """Page 3 – Segmentation RFM."""
    st.header("Segmentation RFM – Priorisation clients")

    st.markdown(
        """
La segmentation *RFM (Recency, Frequency, Monetary)* permet de *classer les clients*
selon leur comportement d’achat afin de *prioriser les actions marketing* :

- *R (Recency)* : fraîcheur de la dernière commande,  
- *F (Frequency)* : intensité d’achat,  
- *M (Monetary)* : valeur générée.

Cette page répond à la question :  
> “Quels segments de clients dois-je cibler en priorité, et comment se répartit la valeur ?”
"""
    )

    # Nombre total de clients et de segments
    nb_clients = rfm.index.nunique()
    nb_segments = rfm["Segment"].nunique()
    ca_total_rfm = rfm["Monetary"].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Clients segmentés (RFM)", f"{nb_clients:,}")
    with c2:
        st.metric("Nombre de segments RFM", nb_segments)
    with c3:
        st.metric("Chiffre d'affaires cumulé (RFM)", f"{ca_total_rfm:,.0f} £")

    st.markdown("---")

    # ---------- Agrégations par segment ----------
    st.subheader("Vue synthétique par segment")

    seg = (
        rfm.groupby("Segment")
        .agg(
            nb_clients=("Recency", "size"),
            recency_moy=("Recency", "mean"),
            freq_moy=("Frequency", "mean"),
            monetary_moy=("Monetary", "mean"),
        )
    )

    # marge et panier moyen approximatifs
    seg["marge_moy"] = seg["monetary_moy"] * 0.40
    seg["panier_moyen"] = seg["monetary_moy"] / seg["freq_moy"]

    seg = seg.sort_values("monetary_moy", ascending=False)

    st.dataframe(seg)

    with st.expander("ℹ Comment lire ce tableau ?"):
        st.write(
            """
- *nb_clients* : combien de clients dans chaque segment  
- *recency_moy* : plus la valeur est *basse*, plus les clients ont commandé récemment  
- *freq_moy* : nombre moyen de commandes  
- *monetary_moy* : valeur client moyenne (CLV observée)  
- *marge_moy* : marge moyenne estimée (ici 40 % du CA)  
- *panier_moyen* : CA moyen par commande  

On peut repérer :
- les segments *très contributeurs* (monetary_moy / marge_moy élevés),
- les segments *en danger* (recency_moy élevée, freq/monetary faibles).
"""
        )

    st.markdown("---")

    # ---------- Graphique : nombre de clients par segment ----------
    st.subheader("Répartition des clients par segment")

    seg_counts = seg["nb_clients"].sort_values(ascending=False)
    st.bar_chart(seg_counts)

    with st.expander("ℹ Utilisation business"):
        st.write(
            """
Ce graphe montre quels segments contiennent le plus de clients.
On peut le croiser avec la valeur moyenne (monetary_moy) pour choisir
où investir du temps et du budget marketing.
"""
        )

    st.markdown("---")

    # ---------- Scatterplot : valeur vs fréquence ----------
    st.subheader("Valeur client vs fréquence d'achat")

    st.markdown(
        "On projette ici chaque client dans le plan *Frequency × Monetary*, "
        "avec une couleur par segment."
    )

    if len(rfm) > 5000:
        rfm_sample = rfm.sample(5000, random_state=42)
    else:
        rfm_sample = rfm.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        rfm_sample["Frequency"],
        rfm_sample["Monetary"],
        c=rfm_sample["RFM_score"],
        cmap=BLUE_SEQUENTIAL,
        s=36,
        alpha=0.8,
        edgecolors=PALETTE["surface"],
        linewidths=0.4,
    )
    ax.set_xlabel("Frequency (nb de commandes)")
    ax.set_ylabel("Monetary (montant total £)")
    ax.set_title("Dispersion des clients dans l’espace RFM", color=PALETTE["ink"], fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Score RFM", color=PALETTE["ink_secondary"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])
    style_axes(ax)
    fig.tight_layout()

    st.pyplot(fig)

    with st.expander("ℹ Lecture du scatterplot"):
        st.write(
            """
Chaque point = un client.

- Axe X : *Frequency* → clients très à droite = clients qui commandent souvent  
- Axe Y : *Monetary* → clients en haut = clients qui dépensent beaucoup  
- Couleur : *score RFM* (plus la couleur est élevée, plus le client est “intéressant”)  
"""
        )

    st.markdown("---")

    st.subheader("Aperçu de la table RFM (50 premiers clients)")
    st.dataframe(rfm.head(50))


def page_scenarios(
    cohort_avg_rev: pd.DataFrame,
    cohort_retention: pd.DataFrame,
    rfm: pd.DataFrame,
):
    """Page 4 – Scénarios CLV (formule fermée + empirique)."""

    st.header("Scénarios – Simulation de CLV")

    st.markdown(
        r"""
Nous combinons ici deux approches pour la *Customer Lifetime Value (CLV)* :

1. *CLV empirique*  
   → obtenue en sommant le *revenu moyen par âge de cohorte* (M0 → M12).  
   C’est la CLV observée sur les données historiques.

2. *CLV “théorique” (formule fermée)*  

\\[
CLV = \frac{m \cdot r}{1 + d - r}
\\]

avec :
- \(m\) = marge moyenne par client *et par mois*,  
- \(r\) = probabilité de *rétention mensuelle*,  
- \(d\) = taux d’*actualisation mensuel*.

Les *sliders* permettent de tester des *scénarios business* :  
“Que se passe-t-il si on augmente la marge ? si on gagne 5 points de rétention ?”
"""
    )

    # --- CLV empirique baseline (depuis les cohortes) ---
    avg_rev_per_age = cohort_avg_rev.mean(axis=0).fillna(0)
    clv_empirique = avg_rev_per_age.sum()

    # --- Paramètres baselines pour la formule fermée ---
    retention_values = cohort_retention.iloc[:, 1:].stack()
    r_base = float(retention_values.mean())
    r_base = max(0.01, min(0.95, r_base))

    monetary_mean = rfm["Monetary"].mean()
    m_base = (monetary_mean / 12) * 0.40

    d_base = 0.01  # 1 % d'actualisation mensuelle

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Paramètres du scénario")

        marge_pct = st.slider(
            "Marge (% du CA)",
            min_value=10,
            max_value=80,
            value=40,
            step=5,
        )

        remise_pct = st.slider(
            "Remise moyenne (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=5,
            help="Représente une baisse moyenne de la marge (ex : promotions, remises).",
        )

        r = st.slider(
            "Rétention mensuelle r",
            min_value=0.10,
            max_value=0.95,
            value=float(round(r_base, 2)),
            step=0.01,
        )

        d = st.slider(
            "Taux d’actualisation d (mensuel)",
            min_value=0.00,
            max_value=0.30,
            value=d_base,
            step=0.01,
        )

        st.caption(
            f"Valeurs de référence approx. : m_base ≈ {m_base:,.0f} £, "
            f"r_base ≈ {r_base:.2f}, d_base = {d_base:.2f}"
        )

    # --- Calculs CLV formule fermée ---
    # marge ajustée par la remise
    m_scenario = (monetary_mean / 12) * (marge_pct / 100) * (1 - remise_pct / 100)

    def clv_closed(m, r, d):
        return m * r / (1 + d - r)

    clv_closed_baseline = clv_closed(m_base, r_base, d_base)
    clv_closed_scenario = clv_closed(m_scenario, r, d)

    delta_clv_abs = clv_closed_scenario - clv_closed_baseline
    delta_clv_pct = (clv_closed_scenario / clv_closed_baseline - 1) * 100

    with col_right:
        st.subheader("Résultats CLV")

        st.metric("CLV empirique (cohortes)", f"{clv_empirique:,.0f} £")
        st.metric("CLV (formule fermée) – baseline", f"{clv_closed_baseline:,.0f} £")
        st.metric(
            "CLV (formule fermée) – scénario",
            f"{clv_closed_scenario:,.0f} £",
            f"{delta_clv_pct:,.1f} %",
        )

        with st.expander("ℹ Interprétation rapide"):
            st.write(
                """
- La *CLV empirique* est basée sur les revenus réels des cohortes.  
- La *CLV formule fermée* permet de tester des scénarios “what-if” sur la marge,
  la rétention, le taux d’actualisation et les remises.  
- La variation en % indique le *gain (ou perte) de valeur client* si on arrive
  à améliorer ces paramètres.
                """
            )

        st.write(
            f"Δ CLV (scénario - baseline) : **{delta_clv_abs:,.0f} £** par client en moyenne."
        )

    st.markdown("---")

    # --- Courbe de sensibilité : CLV en fonction de r ---
    st.subheader("Courbe de sensibilité de la CLV en fonction de r")

    r_values = pd.Series([x / 100 for x in range(10, 96)])
    clv_values = clv_closed(m_scenario, r_values, d)

    sensitivity_df = pd.DataFrame({"r": r_values, "CLV": clv_values}).set_index("r")
    st.line_chart(sensitivity_df)

    with st.expander("ℹ Lecture de la courbe de sensibilité"):
        st.write(
            """
Cette courbe montre comment la CLV théorique varie en fonction de la *rétention mensuelle r*,
pour la marge (%), la remise (%) et le taux d’actualisation fixés par les sliders.
"""
        )


def page_export(rfm: pd.DataFrame):
    """Page 5 – Export (plan d'action)."""
    st.header("Export – Liste de clients activables")

    st.markdown(
        """
Cette page génère une *liste de clients activables* à destination de l’équipe CRM.

Colonnes exportées :

- Customer ID : identifiant unique du client,
- Segment (RFM) : position du client dans la segmentation,
- Frequency : nombre de commandes sur la période,
- Monetary : montant total observé,
- CLV : approximation de la valeur client basée sur Monetary.
"""
    )

    export_df = rfm.reset_index().rename(columns={"index": "Customer ID"}).copy()
    export_df["CLV"] = export_df["Monetary"]

    cols = ["Customer ID", "Segment", "Frequency", "Monetary", "CLV"]
    export_df = export_df[cols]

    st.subheader("Aperçu de la table exportée")
    st.dataframe(export_df.head(50))

    csv = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="💾 Télécharger le CSV des clients activables",
        data=csv,
        file_name="clients_activables.csv",
        mime="text/csv",
    )

    with st.expander("ℹ À mentionner dans le rapport / soutenance"):
        st.write(
            """
- La colonne *CLV* est ici une *approximation* basée sur la valeur observée (Monetary).
- On pourrait la remplacer par une CLV issue de la *formule fermée* en appliquant
  un coefficient commun (par exemple : CLV théorique / Monetary moyen).
- L’idée principale est de fournir au CRM une base exploitable avec Customer ID
  et Segment afin de cibler les campagnes.
            """
        )


# ---------- SIDEBAR : NAVIGATION + FILTRES GLOBAUX ----------


def main():
    col_date, min_date, max_date = get_date_bounds(df_clean)

    with st.sidebar:
        st.title("⚙ Filtres globaux")

        start_date, end_date = st.date_input(
            "Période d'analyse",
            value=(min_date, max_date),
        )

        countries = sorted(df_clean["Country"].dropna().unique().tolist())
        selected_countries = st.multiselect(
            "Pays",
            options=countries,
            default=countries,
        )

        returns_mode = st.selectbox(
            "Retours (factures 'C')",
            ["Inclure", "Exclure", "Neutraliser"],
        )

        min_amount = st.number_input(
            "Seuil minimum de montant par transaction (Amount)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Overview", "Cohortes", "Segmentation RFM", "Scénarios", "Export"],
        )

    # Application des filtres sur les transactions
    df_filtered = filter_transactions(
        df_clean,
        col_date=col_date,
        start_date=start_date,
        end_date=end_date,
        countries=selected_countries,
        returns_mode=returns_mode,
        min_amount=min_amount,
    )

    # Affichage du résumé des filtres actifs
    st.caption(
        f"Filtres actifs — période: {start_date} → {end_date} | "
        f"pays: {len(selected_countries)} sélectionnés | "
        f"retours: {returns_mode} | "
        f"seuil montant: {min_amount:.0f} £ | "
        f"n={len(df_filtered):,} lignes"
    )

    # Routing des pages
    if page == "Overview":
        page_overview(df_clean, df_filtered, cohort_avg_rev, rfm)
    elif page == "Cohortes":
        page_cohortes(cohort_retention, cohort_avg_rev)
    elif page == "Segmentation RFM":
        page_rfm(rfm)
    elif page == "Scénarios":
        page_scenarios(cohort_avg_rev, cohort_retention, rfm)
    elif page == "Export":
        page_export(rfm)


if __name__ == "__main__":
    main()
