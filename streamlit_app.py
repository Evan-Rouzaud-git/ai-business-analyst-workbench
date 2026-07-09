# streamlit_app.py
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # cache les barres de téléchargement du hub

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="transformers")  # supprime les warnings d'imports vision

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from src.prediction import generate_forecast
import transformers
transformers.logging.set_verbosity_error()  # désactive les logs d'info de transformers

st.set_page_config(
    page_title="AI Business Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #cccccc; margin-bottom: 1rem; }
    .kpi-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #4a4a6a;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #ffffff; }
    .kpi-label { font-size: 0.9rem; color: #aaaaaa; }
    .priority-card {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 5px solid;
        background-color: #1e1e2e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .priority-high { border-left-color: #dc3545; }
    .priority-medium { border-left-color: #ffc107; }
    .priority-low { border-left-color: #28a745; }
    .stButton button { width: 100%; }
    .streamlit-expanderHeader { color: #ffffff !important; }
    .streamlit-expanderContent { color: #dddddd !important; }
    .theme-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border: 1px solid #3a3a5a;
    }
    .theme-title { font-size: 1.1rem; font-weight: 600; color: #ffffff; }
    .theme-stats { font-size: 0.9rem; color: #bbbbbb; }
    .theme-share { font-size: 0.8rem; color: #8888aa; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'cluster_info' not in st.session_state:
    st.session_state.cluster_info = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'total_tickets' not in st.session_state:
    st.session_state.total_tickets = 0
if 'total_gain' not in st.session_state:
    st.session_state.total_gain = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'risk_scores' not in st.session_state:
    st.session_state.risk_scores = None
if 'hourly_rate' not in st.session_state:
    st.session_state.hourly_rate = 25
if 'default_reduction' not in st.session_state:
    st.session_state.default_reduction = 0.5  # Changé de 0.3 à 0.5 pour correspondre aux options du slider

# Sidebar
st.sidebar.title("📂 Dataset")
dataset_option = st.sidebar.radio(
    "Choisir un dataset",
    ["Dataset test (1000 lignes)", "Upload CSV"]
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = None
if dataset_option == "Dataset test (1000 lignes)":
    try:
        df = load_data("data/data.csv")
        st.sidebar.success("✅ Dataset chargé")
    except FileNotFoundError:
        st.sidebar.error("Fichier data.csv introuvable. Veuillez uploader un fichier.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Fichier chargé")

if df is None:
    st.warning("Veuillez charger un dataset pour commencer.")
    st.stop()

# Bouton Analyser
# ... (début du fichier inchangé) ...

# Bouton Analyser
if st.sidebar.button("🚀 Analyser", type="primary", use_container_width=True):
    from src.clustering import ClusteringEngine

    st.session_state.analysis_done = False

    # Création du placeholder pour la barre de progression
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0.0, text="Démarrage...")
    
    # Statut avec plus de place
    status = st.status("🔄 Analyse en cours...", expanded=True)

    # Fonction callback améliorée
    def clustering_progress(msg, progress):
        # Mise à jour de la barre
        progress_bar.progress(min(progress, 0.95), text=msg)  # On limite à 95% pour laisser place aux étapes finales
        status.write(f"  {msg}")
        status.update(label=msg)

    try:
        # Étape 1 : Préparation (0% → 5%)
        progress_bar.progress(0.0, text="🧹 Préparation des données...")
        status.update(label="🧹 Préparation des données...")
        status.write("  📋 Nettoyage et mise en forme du dataset")
        
        df_clean = df.copy()
        df_clean['description'] = df_clean['description'].fillna('').astype(str)
        if 'Date' in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean['text_length'] = df_clean['description'].str.len()
        if 'has_response' in df_clean.columns:
            df_clean['has_response'] = df_clean['has_response'].astype(int)
        
        progress_bar.progress(0.05, text="✅ Données préparées")

        # Étape 2 : Clustering (5% → 95%, géré par le callback)
        status.update(label="🧠 Analyse des thèmes...")
        engine = ClusteringEngine(n_macro=6, n_clusters_fin=30)
        labels = engine.fit(df_clean, text_column='description', progress_callback=clustering_progress)
        cluster_info = engine.get_cluster_info()
        
        # Stockage en session
        st.session_state.df_clean = df_clean
        st.session_state.engine = engine
        st.session_state.cluster_info = cluster_info
        st.session_state.labels = labels
        st.session_state.embeddings = engine.embeddings

        # Étape 3 : Métriques (95% → 97%)
        progress_bar.progress(0.95, text="📊 Calcul des métriques...")
        status.update(label="📊 Calcul des métriques...")
        status.write("  💰 Estimation des coûts et gains")
        
        total_tickets = len(df_clean)
        hourly_rate = st.session_state.hourly_rate
        default_duree = 15
        default_reduction = st.session_state.default_reduction

        total_cost = 0
        total_gain = 0
        for cluster in cluster_info:
            cout_unitaire = (default_duree / 60) * hourly_rate
            cout_total = cluster['count'] * cout_unitaire
            gain = cout_total * default_reduction
            total_cost += cout_total
            total_gain += gain

        st.session_state.total_tickets = total_tickets
        st.session_state.total_gain = total_gain
        st.session_state.total_cost = total_cost
        progress_bar.progress(0.97, text="📊 Métriques calculées")

        # Étape 4 : Prédictions (97% → 99%)
        progress_bar.progress(0.97, text="🔮 Prédictions...")
        status.update(label="🔮 Prédictions...")
        status.write("  📈 Génération des prévisions")
        
        forecast = generate_forecast(df_clean, date_column='Date')
        st.session_state.forecast = forecast

        # Risque
        if 'has_response' in df_clean.columns and 'category' in df_clean.columns:
            features = ['category', 'text_length']
            X_rf = df_clean[features].copy()
            le_cat = LabelEncoder()
            X_rf['category_enc'] = le_cat.fit_transform(X_rf['category'].fillna('unknown'))
            X_rf = X_rf[['category_enc', 'text_length']]
            y_rf = df_clean['has_response']
            if len(X_rf) > 10 and y_rf.nunique() > 1:
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_rf, y_rf)
                proba = clf.predict_proba(X_rf)[:, 1]
                risk_score = 1 - proba
                df_clean['risk_score'] = risk_score
                st.session_state.risk_scores = risk_score
            else:
                st.session_state.risk_scores = None
        else:
            st.session_state.risk_scores = None

        st.session_state.analysis_done = True
        
        # 🎉 FIN DE L'ANALYSE - Suppression de la barre et effet "wow"
        progress_placeholder.empty()  # La barre disparaît !
        
        # Mise à jour du statut en mode "succès"
        status.update(
            label="✅ Analyse terminée avec succès !",
            state="complete"
        )
        status.write("  📊 6 thèmes identifiés")
        status.write(f"  💰 {total_gain:,.0f} € de gain potentiel")
        status.write(f"  📈 {total_tickets:,} réclamations analysées")
        
        # Effet "wow" pour la démo
        st.success("✨ Analyse terminée ! Les résultats sont disponibles dans les onglets ci-dessous.")

    except Exception as e:
        # Gestion d'erreur propre
        progress_placeholder.empty()
        status.update(label="❌ Erreur lors de l'analyse", state="error")
        status.write(f"  {str(e)}")
        st.error(f"Une erreur est survenue : {str(e)}")

# ... (suite du code inchangé) ...
# Affichage
if st.session_state.analysis_done and st.session_state.df_clean is not None:
    df_clean = st.session_state.df_clean
    cluster_info = st.session_state.cluster_info
    engine = st.session_state.engine

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Résumé", "📈 Analyse", "📋 Macro-thèmes", "💰 ROI", "🔮 Prédictions"])

    with tab1:
        st.markdown('<div class="main-header">📊 Résumé Exécutif</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Synthèse en 15 secondes</div>', unsafe_allow_html=True)

        total = st.session_state.total_tickets
        total_gain = st.session_state.total_gain
        total_cost = st.session_state.total_cost
        roi = (total_gain / total_cost * 100) if total_cost > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-label">Réclamations</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_cost:,.0f} €</div>
                <div class="kpi-label">Coût annuel estimé</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_gain:,.0f} €</div>
                <div class="kpi-label">Gain potentiel</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🎯 Les 3 priorités à traiter")

        # ------------------------------------------------------------
        # 1. Calcul des critères pour chaque cluster
        # ------------------------------------------------------------
        df = st.session_state.df_clean
        labels = st.session_state.labels
        hourly_rate = st.session_state.hourly_rate
        default_duree = 15  # minutes
        default_reduction = st.session_state.default_reduction

        # Métriques globales pour les seuils
        all_counts = [c['count'] for c in cluster_info]
        all_gains = []
        for c in cluster_info:
            gain = c['count'] * (default_duree / 60) * hourly_rate * 12 * default_reduction
            all_gains.append(gain)

        mean_count = np.mean(all_counts)
        mean_gain = np.mean(all_gains)

        # Longueur médiane globale (si colonne text_length existe)
        if 'text_length' in df.columns:
            median_length = df['text_length'].median()
        else:
            median_length = None

        # Vérifier disponibilité des colonnes
        has_response_col = 'has_response' in df.columns
        has_date_col = 'Date' in df.columns

        # Préparer les données pour la croissance (si date)
        if has_date_col:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # On prend les 3 derniers mois pour la tendance
            date_max = df['Date'].max()
            date_min_3m = date_max - pd.DateOffset(months=3)
            mask_recent = df['Date'] >= date_min_3m

        # Stocker les critères activés par cluster
        cluster_criteria = []

        for cluster in cluster_info:
            cid = cluster['cluster_id']
            # indices des tickets du cluster
            idx = np.where(labels == cid)[0]
            count = len(idx)
            gain = count * (default_duree / 60) * hourly_rate * 12 * default_reduction

            # 1. Volume élevé
            crit_volume = count > mean_count

            # 2. Gain important
            crit_gain = gain > mean_gain

            # 3. Non-réponse élevée (si disponible)
            crit_no_response = False
            if has_response_col:
                resp_rate = df.iloc[idx]['has_response'].mean()  # 1 = a réponse, 0 = non
                no_resp_rate = 1 - resp_rate
                global_no_resp = 1 - df['has_response'].mean()
                crit_no_response = no_resp_rate > global_no_resp

            # 4. Croissance récente (si disponible)
            crit_growth = False
            if has_date_col:
                sub_df = df.iloc[idx][mask_recent.iloc[idx]]  # tickets récents du cluster
                if len(sub_df) > 5:  # assez de points pour une régression
                    # Regrouper par mois
                    sub_df = sub_df.copy()  # éviter les warning
                    sub_df['month'] = sub_df['Date'].dt.to_period('M')
                    monthly = sub_df.groupby('month').size().reset_index(name='count')
                    if len(monthly) >= 3:  # au moins 3 mois
                        # Trier par mois chronologique
                        monthly = monthly.sort_values('month')
                        # Créer un index numérique simple (0,1,2,...)
                        monthly['month_num'] = range(len(monthly))
                        # Régression linéaire
                        X = monthly['month_num'].values.reshape(-1, 1)
                        y = monthly['count'].values
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(X, y)
                        slope = reg.coef_[0]
                        crit_growth = slope > 0

            # 5. Facilité de résolution (longueur courte)
            crit_easy = False
            if median_length is not None:
                mean_len = df.iloc[idx]['text_length'].mean()
                crit_easy = mean_len < median_length

            # Compter les critères activés
            active_criteria = [
                ('📈 Volume élevé', crit_volume),
                ('💰 Gain important', crit_gain),
                ('⏳ Non-réponse élevée', crit_no_response),
                ('📊 Croissance récente', crit_growth),
                ('🧩 Facile à résoudre', crit_easy)
            ]
            # Filtrer ceux qui sont True
            reasons = [label for label, flag in active_criteria if flag]
            score = len(reasons)

            cluster_criteria.append({
                'cluster_id': cid,
                'count': count,
                'share': count / len(df) * 100,
                'gain': gain,
                'score': score,
                'reasons': reasons,
                'go': score >= 2  # GO si au moins 3 critères
            })

        # Trier par score décroissant, puis par gain
        sorted_clusters = sorted(cluster_criteria, key=lambda x: (x['score'], x['gain']), reverse=True)[:3]

        # ------------------------------------------------------------
        # 2. Affichage des 3 priorités
        # ------------------------------------------------------------
        cols = st.columns(3)
        for i, item in enumerate(sorted_clusters):
            with cols[i]:
                nom = engine.noms_sujets.get(item['cluster_id'], f"Thème {item['cluster_id']}")
                # Couleur de la bordure selon score
                if item['score'] >= 4:
                    border_class = "priority-high"
                elif item['score'] >= 3:
                    border_class = "priority-medium"
                else:
                    border_class = "priority-low"

                go_badge = "🟢 GO" if item['go'] else "🔴 NO GO"

                # Construction de la chaîne des raisons
                reasons_text = "<br>".join([f"• {r}" for r in item['reasons']]) if item['reasons'] else "Aucun critère activé"

                st.markdown(f"""
                <div class="priority-card {border_class}">
                    <strong>{nom}</strong><br>
                    {item['count']} réclamations ({item['share']:.1f}%)<br>
                    Gain estimé: {item['gain']:,.0f} €<br><br>
                    <span style="font-size:0.9rem; color:#aaa;">Priorité : {item['score']}/5 critères</span><br>
                    <span style="font-size:1.1rem; font-weight:bold;">{go_badge}</span><br><br>
                    <span style="font-size:0.9rem;"><strong>Pourquoi ?</strong></span><br>
                    {reasons_text}
                </div>
                """, unsafe_allow_html=True)

        # Message de recommandation
        if sorted_clusters:
            top = sorted_clusters[0]
            nom = engine.noms_sujets.get(top['cluster_id'], f"Thème {top['cluster_id']}")
            st.markdown("<br>", unsafe_allow_html=True) 
            st.success(f"💡 **Si une seule action devait être menée ce trimestre, nous recommandons d'agir sur « {nom} », car il active {top['score']} critères sur 5 et représente {top['share']:.1f}% des réclamations.**")


            # ---- Nouvelle section : Matrice de priorisation ----
            st.markdown("---")
            st.subheader("🎯 Matrice de priorisation par catégorie")

            # Vérifier que les colonnes nécessaires existent
            df = st.session_state.df_clean
            if all(col in df.columns for col in ['category', 'has_response', 'status']):
                # 1. Agrégation par catégorie
                cluster_data = df.groupby('category').agg(
                    nb_signalements=('category', 'size'),
                    taux_non_reponse=('has_response', lambda x: (x == 0).mean()),
                    taux_non_traite=('status', lambda x: (x == 'NonTraite').mean())
                ).reset_index().rename(columns={'category': 'projet'})

                # Complexité (sous-catégories uniques si disponible)
                if 'subcategory' in df.columns:
                    complexite = df.groupby('category')['subcategory'].nunique().reset_index()
                    complexite.columns = ['projet', 'complexite']
                    cluster_data = cluster_data.merge(complexite, on='projet', how='left')
                    cluster_data['complexite'] = cluster_data['complexite'].fillna(1)
                else:
                    cluster_data['complexite'] = 1

                # 2. Impact normalisé
                cluster_data['impact_raw'] = (cluster_data['nb_signalements'] *
                                            (cluster_data['taux_non_reponse'] + cluster_data['taux_non_traite']) / 2)
                min_impact = cluster_data['impact_raw'].min()
                max_impact = cluster_data['impact_raw'].max()
                cluster_data['impact'] = (cluster_data['impact_raw'] - min_impact) / (max_impact - min_impact) if max_impact > min_impact else 0.5

                # 3. Effort normalisé
                max_nb = cluster_data['nb_signalements'].max()
                max_comp = cluster_data['complexite'].max()
                if max_nb > 0:
                    cluster_data['effort_raw'] = (cluster_data['nb_signalements'] / max_nb) * 0.7 + (cluster_data['complexite'] / max_comp) * 0.3 if max_comp > 0 else (cluster_data['nb_signalements'] / max_nb)
                else:
                    cluster_data['effort_raw'] = 0
                min_effort = cluster_data['effort_raw'].min()
                max_effort = cluster_data['effort_raw'].max()
                cluster_data['effort'] = (cluster_data['effort_raw'] - min_effort) / (max_effort - min_effort) if max_effort > min_effort else 0.5

                # 4. Score de priorité
                cluster_data['priorite_score'] = cluster_data['impact'] * (1 - cluster_data['effort'])

                # 5. Graphique
                cluster_data['projet_court'] = cluster_data['projet'].str.replace('Achat', '').str.replace('Internet', 'Web')

                fig = px.scatter(cluster_data,
                                x='effort', y='impact',
                                size='nb_signalements',
                                color='priorite_score',
                                text='projet_court',
                                title='🎯 Matrice de priorisation : Impact vs Effort',
                                labels={'effort': 'Effort estimé',
                                        'impact': 'Impact potentiel',
                                        'nb_signalements': 'Volume',
                                        'priorite_score': 'Priorité'},
                                color_continuous_scale='RdYlGn_r',
                                size_max=40,
                                height=500)
                fig.update_traces(
                    textposition='top center',
                    textfont=dict(size=10),
                    marker=dict(line=dict(width=1, color='white'), opacity=0.85),
                    hovertemplate='<b>%{text}</b><br>Impact: %{y:.2f}<br>Effort: %{x:.2f}<br>Volume: %{marker.size}<br>Score: %{marker.color:.2f}<extra></extra>'
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4, line_width=3)
                fig.add_vline(x=0.5, line_dash="dash", line_color="white", opacity=0.4, line_width=3)
                fig.add_annotation(x=0.15, y=0.92, text="🚀 Quick Wins", showarrow=False, font=dict(size=16, color="green"))
                fig.add_annotation(x=0.85, y=0.92, text="💎 Projets Majeurs", showarrow=False, font=dict(size=16, color="blue"))
                fig.add_annotation(x=0.15, y=0.08, text="⚡ Petits Gains", showarrow=False, font=dict(size=16, color="orange"))
                fig.add_annotation(x=0.85, y=0.08, text="❓ À réévaluer", showarrow=False, font=dict(size=16, color="red"))
                fig.update_layout(
                    xaxis=dict(range=[-0.1, 1.1], showgrid=True, gridwidth=0.5),
                    yaxis=dict(range=[-0.1, 1.1], showgrid=True, gridwidth=0.5),
                    coloraxis_colorbar=dict(title="Priorité"),
                    hoverlabel=dict(font_size=11),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---- Heatmap des actions ----
                st.markdown("---")
                st.subheader("🌡️ Matrice des actions par catégorie")

                actions_data = df.groupby('category').agg(
                    has_response=('has_response', lambda x: (x == 0).mean() * 100),
                    status=('status', lambda x: (x == 'NonTraite').mean() * 100),
                    nb_signalements=('category', 'size')
                ).reset_index()

                def priorite(row):
                    if row['has_response'] > 50 and row['nb_signalements'] > 100:
                        return '🔴 CRITIQUE'
                    elif row['has_response'] > 30 or row['nb_signalements'] > 200:
                        return '🟡 IMPORTANT'
                    else:
                        return '🟢 SURVEILLER'
                actions_data['priorite'] = actions_data.apply(priorite, axis=1)

                heatmap_data = actions_data[['category', 'has_response', 'status']].set_index('category')
                if 'is_read' in df.columns:
                    is_read = df.groupby('category')['is_read'].apply(lambda x: (x == 0).mean() * 100).reset_index()
                    is_read.columns = ['category', 'is_read']
                    heatmap_data = heatmap_data.merge(is_read.set_index('category'), left_index=True, right_index=True)
                    heatmap_data = heatmap_data[['has_response', 'is_read', 'status']]
                heatmap_data.rename(columns={
                    'has_response': 'Sans réponse (%)',
                    'status': 'Non traité (%)',
                    'is_read': 'Non lu (%)'
                }, inplace=True)
                fig2 = px.imshow(heatmap_data.T,
                                text_auto='.1f',
                                title='🌡️ Matrice des actions par catégorie',
                                labels=dict(x="Catégorie", y="Métrique", color="%"),
                                color_continuous_scale='RdYlGn_r',
                                template=None, 
                                height=400)
   
                st.plotly_chart(fig2, use_container_width=True)

                with st.expander("📋 Détail des priorités d'action par catégorie"):
                    st.dataframe(actions_data[['category', 'nb_signalements', 'has_response', 'status', 'priorite']],
                                use_container_width=True, hide_index=True)

            else:
                st.info("ℹ️ Les colonnes nécessaires ('category', 'has_response', 'status') ne sont pas toutes disponibles. Matrice non affichée.")
    with tab2:
        st.markdown('<div class="main-header">📈 Analyse des réclamations</div>', unsafe_allow_html=True)

        # ---- 1er graphique : Répartition par catégorie ----
        if 'category' in df_clean.columns:
            cat_counts = df_clean['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            fig = px.bar(
                cat_counts,
                x='category',
                y='count',
                title="Répartition des tickets par catégorie",
                color='category',
                text='count'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'category' absente – graphique non disponible.")

        # ---- 2ème graphique : Évolution mensuelle du nombre total ----
        if 'Date' in df_clean.columns and not df_clean['Date'].isna().all():
            df_clean['month'] = df_clean['Date'].dt.to_period('M')
            monthly = df_clean.groupby('month').size().reset_index(name='count')
            monthly['month'] = monthly['month'].astype(str)
            fig = px.line(
                monthly,
                x='month',
                y='count',
                title="Évolution mensuelle du nombre total de tickets",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'Date' absente ou invalide – graphique non disponible.")

        # ---- NOUVEAU : Évolution HEBDOMADAIRE des catégories de statuts ----
        if 'status' in df_clean.columns and 'Date' in df_clean.columns:
            # Mapping des statuts
            categories = {
                'À traiter': ['NonTraite', 'NonConsulte', 'ConsulteIgnore'],
                'En cours': ['TraitementEnCours', 'PromesseAction'],
                'Clos': ['Infonde', 'MalAttribue', 'Transmis'],
                'RGPD': ['SuppressionRGPD'],
                'Information': ['InformateurInterne']
            }
            df_clean['categorie_statut'] = df_clean['status'].map(
                {statut: cat for cat, statuts in categories.items() for statut in statuts}
            )

            # Regroupement par semaine (on utilise le début de semaine, ex: lundi)
            df_clean['semaine'] = df_clean['Date'].dt.to_period('W').dt.start_time
            data_cat_par_semaine = df_clean.groupby(
                [df_clean['semaine'], 'categorie_statut']
            ).size().unstack(fill_value=0)
            data_cat_par_semaine.index = data_cat_par_semaine.index.astype(str)  # pour l'affichage

            # Tracer avec Plotly
            fig = px.line(
                data_cat_par_semaine,
                x=data_cat_par_semaine.index,
                y=data_cat_par_semaine.columns,
                title="Évolution hebdomadaire des catégories de statuts",
                markers=True,
                labels={'value': 'Nombre de requêtes', 'variable': 'Catégorie de statut'}
            )
            fig.update_layout(
                xaxis_title='Semaine (début)',
                yaxis_title='Nombre de requêtes',
                legend_title='Catégorie',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne 'status' ou 'Date' manquante – graphique non disponible.")

        # ---- NOUVEAU : Top 10 des tags (avec couleurs contrastées) ----
        if 'tags' in df_clean.columns:
            all_tags = df_clean['tags'].str.split(',').explode().str.strip()
            all_tags = all_tags[all_tags != 'aucun_tag']
            tag_counts = all_tags.value_counts().head(10).reset_index()
            tag_counts.columns = ['tag', 'count']

            if not tag_counts.empty:
                # On utilise une couleur fixe orange ou une palette contrastée
                fig = px.bar(
                    tag_counts,
                    x='count',
                    y='tag',
                    orientation='h',
                    title='Top 10 des tags',
                    labels={'count': 'Nombre', 'tag': 'Tag'},
                    color='count',
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title='Nombre',
                    yaxis_title='Tag',
                )
                # Optionnel : on peut aussi forcer une couleur unique en désactivant color
                # fig = px.bar(..., color_discrete_sequence=['#FF8C00']) 
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun tag valide trouvé (ou tous sont 'aucun_tag').")
        else:
            st.info("Colonne 'tags' absente – graphique non disponible.")

    with tab3:
        st.markdown('<div class="main-header">📋 Macro-thèmes</div>', unsafe_allow_html=True)
        sorted_clusters = sorted(cluster_info, key=lambda x: x['count'], reverse=True)
        cols = st.columns(min(2, len(sorted_clusters)))
        for i, cluster in enumerate(sorted_clusters):
            col = cols[i % 2]
            with col:
                nom = engine.noms_sujets.get(cluster['cluster_id'], f"Thème {cluster['cluster_id']}")
                mots_cles = ', '.join(cluster['top_words'][:3])
                with st.container():
                    st.markdown(f"""
                    <div class="theme-card">
                        <div class="theme-title">🔹 {nom}</div>
                        <div class="theme-stats">📌 {cluster['count']} réclamations • {cluster['share']:.1f}%</div>
                        <div class="theme-stats">🏷️ {mots_cles}</div>
                        <div class="theme-share">Catégories : {', '.join([f'{k} ({v})' for k, v in cluster['categories'].items()])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Voir les exemples"):
                        for ex in cluster['examples'][:3]:
                            st.markdown(f"- {ex}")

    with tab4:
        st.markdown('<div class="main-header">💰 Simulateur ROI par cluster</div>', unsafe_allow_html=True)

        hourly_rate = st.number_input("Coût horaire moyen d'un agent (€)", value=st.session_state.hourly_rate, min_value=10, step=1, key="roi_hourly_rate")
        st.session_state.hourly_rate = hourly_rate

        st.markdown("---")
        st.subheader("Paramètres par thème")

        reductions = {}
        temps_moyens = {}

        cols = st.columns(min(3, len(cluster_info)))
        for i, cluster in enumerate(cluster_info):
            col = cols[i % 3]
            with col:
                nom = engine.noms_sujets.get(cluster['cluster_id'], f"Thème {cluster['cluster_id']}")
                st.markdown(f"**{nom}**")
                default_val = int(st.session_state.default_reduction * 100)
                options = [0, 20, 50, 80, 100]
                if default_val not in options:
                    default_val = 50
                reduction = st.select_slider(
                    "Réduction potentielle",
                    options=options,
                    value=default_val,
                    key=f"reduc_{cluster['cluster_id']}"
                )
                reductions[cluster['cluster_id']] = reduction / 100.0

                temps = st.number_input(
                    "Temps moyen (min)",
                    min_value=1,
                    max_value=120,
                    value=15,
                    step=1,
                    key=f"temps_{cluster['cluster_id']}"
                )
                temps_moyens[cluster['cluster_id']] = temps

        st.markdown("---")
        st.subheader("📊 Calculs détaillés")

        rows = []
        total_cout = 0
        total_gain = 0
        for cluster in cluster_info:
            cid = cluster['cluster_id']
            volume = cluster['count']
            duree = temps_moyens.get(cid, 15)
            reduction = reductions.get(cid, 0.5)
            nom = engine.noms_sujets.get(cid, f"Thème {cid}")

            cout_unitaire = (duree / 60) * hourly_rate
            cout_total = volume * cout_unitaire
            gain = cout_total * reduction
            total_cout += cout_total
            total_gain += gain

            rows.append({
                "Thème": nom,
                "Volume": volume,
                "Tps (min)": duree,
                "Coût unitaire": f"{cout_unitaire:.2f} €",
                "Coût total": f"{cout_total:.2f} €",
                "Réduction": f"{reduction*100:.0f}%",
                "Gain": f"{gain:.2f} €",
                "gain_value": gain  # colonne numérique pour tri et graphique
            })

        df_roi = pd.DataFrame(rows)

        # Mise à jour des totaux dans la session
        st.session_state.total_cost = total_cout
        st.session_state.total_gain = total_gain

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Coût total actuel", f"{total_cout:,.2f} €")
        with col2:
            st.metric("Gain total potentiel", f"{total_gain:,.2f} €")

        roi = (total_gain / total_cout * 100) if total_cout > 0 else 0
        st.metric("ROI estimé", f"{roi:.1f}%", help="ROI = (Gain / Coût) × 100")

        # Graphique des gains trié par gain décroissant (utilise la colonne numérique)
        df_roi_sorted = df_roi.sort_values(by="gain_value", ascending=False)
        fig = px.bar(
            df_roi_sorted,
            x="Thème",
            y="gain_value",
            title="Gain potentiel par thème",
            text=[f"{g:,.0f} €" for g in df_roi_sorted["gain_value"]],
            color="Thème"
        )
        fig.update_traces(textposition='outside', textfont_size=11)
        fig.update_layout(
            yaxis_title="Gain (€)",
            xaxis_title="",
            yaxis=dict(tickformat=",.0f")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Affichage du tableau détaillé (sans la colonne numérique)
        st.dataframe(df_roi.drop(columns=["gain_value"]), use_container_width=True, hide_index=True)

    with tab5:
        st.markdown('<div class="main-header">🔮 Prédictions et alertes</div>', unsafe_allow_html=True)

        forecast_data = st.session_state.forecast
        if forecast_data is None:
            st.info("🔮 Prévision non disponible (pas assez de données temporelles).")
        elif not all(k in forecast_data for k in ('historical', 'current_week', 'forecast')):
            st.error("""
                ⚠️ **Version incompatible de `prediction.py`**  
                Veuillez mettre à jour le fichier `prediction.py` avec la nouvelle version.
            """)
        else:
            st.subheader("📈 Prévision des réclamations")

            # Récupération et normalisation
            df_hist = forecast_data['historical'].copy()
            df_cur = forecast_data['current_week'].copy()   # dernière semaine complète + semaine en cours
            df_pred = forecast_data['forecast'].copy()      # dernière semaine complète + 4 semaines futures

            for df in (df_hist, df_cur, df_pred):
                df['ds'] = pd.to_datetime(df['ds']).dt.normalize()

            # --- Graphique ---
            fig = go.Figure()

            # 1. Historique (bleu)
            if not df_hist.empty:
                fig.add_trace(go.Scatter(
                    x=df_hist['ds'], y=df_hist['y'],
                    mode='lines+markers',
                    name='Historique',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=8)
                ))

            # 2. Semaine en cours (vert) : dernière semaine complète + semaine en cours (partielle)
            if not df_cur.empty:
                fig.add_trace(go.Scatter(
                    x=df_cur['ds'], y=df_cur['y'],
                    mode='lines+markers',
                    name='Semaine en cours',
                    line=dict(color='#2ca02c', width=2, dash='dot'),
                    marker=dict(symbol='diamond', size=10, color='#2ca02c')
                ))

            # 3. Prévision (orange) : dernière semaine complète + 4 semaines futures
            if not df_pred.empty:
                fig.add_trace(go.Scatter(
                    x=df_pred['ds'], y=df_pred['y'],
                    mode='lines+markers',
                    name='Prévision (4 semaines)',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(symbol='triangle-up', size=10, color='#ff7f0e')
                ))

            # Ligne verticale aujourd'hui (pour info, on peut la garder)
            today = pd.Timestamp.now().normalize()
            fig.add_vline(x=today, line_width=1.5, line_dash="dash", line_color="gray", opacity=0.7)

            # --- Axe X : un tick par lundi (numéro de semaine) ---
            all_dates = pd.Series()
            for df in (df_hist, df_cur, df_pred):
                if not df.empty:
                    all_dates = pd.concat([all_dates, df['ds']])

            if not all_dates.empty:
                all_dates = all_dates.sort_values().unique()
                mondays = [d for d in all_dates if d.weekday() == 0]
                labels = [f"S{d.isocalendar().week}" for d in mondays]
                xaxis_tickvals = mondays
                xaxis_ticktext = labels
            else:
                xaxis_tickvals = None
                xaxis_ticktext = None

            fig.update_layout(
                title="📊 Évolution hebdomadaire – historique, semaine en cours et prévision",
                xaxis_title="Semaine",
                yaxis_title="Nombre de réclamations",
                hovermode='x unified',
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="left", x=0.01,
                    bgcolor='rgba(0,0,0,0)'
                ),
                xaxis=dict(
                    tickvals=xaxis_tickvals,
                    ticktext=xaxis_ticktext,
                    tickangle=0
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- KPI ---
            avg_hist = df_hist['y'].mean() if not df_hist.empty else 0

            # Première semaine prédite (après la connexion)
            if len(df_pred) > 1:
                next_week_val = df_pred['y'].iloc[1]
            else:
                next_week_val = 0

            # Dernière semaine de la prévision (pic potentiel)
            if len(df_pred) > 0:
                last_week_val = df_pred['y'].iloc[-1]
            else:
                last_week_val = 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Moyenne historique (hebdo)", f"{int(avg_hist)}")
            with col2:
                st.metric("🔮 Prévision semaine prochaine", f"{int(next_week_val)}",
                        delta=f"{int(next_week_val - avg_hist)} vs moyenne")
            with col3:
                if last_week_val > avg_hist * 1.2 and avg_hist > 0:
                    st.metric("⚠️ Prévision dernière semaine", f"{int(last_week_val)}", delta="Alerte pic")
                else:
                    st.metric("📈 Prévision dernière semaine", f"{int(last_week_val)}")

            # --- Alertes ---
            # Informations sur la semaine en cours
            if len(df_cur) > 1:  # on a la dernière semaine complète + la semaine en cours
                # La semaine en cours est la dernière ligne
                current_week = df_cur.iloc[-1]
                # Vérifier que c'est bien la semaine actuelle (pour éviter les confusions)
                current_week_start = today - pd.Timedelta(days=today.weekday())
                if current_week['ds'] == current_week_start:
                    st.info(f"📊 **Semaine en cours** : {int(current_week['y'])} réclamations (du {current_week['ds'].strftime('%d/%m')} à aujourd'hui).")
                    
                    # Comparaison avec la moyenne historique
                    if current_week['y'] > avg_hist * 1.5 and avg_hist > 0:
                        st.warning(f"⚠️ La semaine en cours est déjà {int((current_week['y']/avg_hist - 1)*100)}% au-dessus de la moyenne historique !")
                    elif current_week['y'] < avg_hist * 0.5 and avg_hist > 0:
                        st.info(f"📉 La semaine en cours est {int((1 - current_week['y']/avg_hist)*100)}% en dessous de la moyenne historique.")

            # Alerte sur le pic prévu
            if last_week_val > avg_hist * 1.2 and avg_hist > 0:
                last_date = df_pred['ds'].iloc[-1]
                date_str = last_date.strftime('%d/%m')
                st.warning(f"⚠️ **Pic prévu** la semaine du {date_str} avec {int(last_week_val)} réclamations (moyenne : {int(avg_hist)}).")
            else:
                st.info(f"📊 **Prévision stable** : ~{int(next_week_val)} réclamations la semaine prochaine.")

            # --- Tickets à risque ---
            st.subheader("⚠️ Tickets à risque de non-réponse")
            if st.session_state.risk_scores is not None:
                df_clean = st.session_state.df_clean
                df_clean['risk'] = st.session_state.risk_scores
                risky_tickets = df_clean.nlargest(10, 'risk')[['description', 'category', 'risk']]
                risky_tickets['risk'] = risky_tickets['risk'].apply(lambda x: f"{x:.0%}")
                st.dataframe(risky_tickets, use_container_width=True)
            else:
                st.info("📋 Pas assez de données pour évaluer le risque.")
 
else:
    if not st.session_state.analysis_done:
        st.info("Cliquez sur 'Analyser' pour démarrer l'analyse.")