import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import mlflow
import os
from datetime import datetime
from pathlib import Path
import plotly.express as px
import streamlit.components.v1 as components
import random
import subprocess
import sys
import io 
# 1. On vérifie et on installe gTTS de force au démarrage
try:
    from gtts import gTTS
except ImportError:
    # Cette commande utilise le flag -m pour viser le bon environnement
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gtts"])
    from gtts import gTTS
import base64


# --- GESTION DE MLFLOW ---
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    # On crée un objet "fantôme" pour éviter que le reste du code ne plante
    class FakeMLflow:
        def log_metric(self, *args, **kwargs): pass
        def set_tag(self, *args, **kwargs): pass
    mlflow = FakeMLflow()

# Maintenant ton "if not HAS_MLFLOW" fonctionnera sans erreur !
if not HAS_MLFLOW:
    st.sidebar.warning("⚠️ Mode local : MLflow n'est pas installé dans ce conteneur.")

class SystemMonitor:
    """
    Elite Agri-Intelligence Monitoring Dashboard.
    Fournit un suivi en temps réel des performances du modèle et une gouvernance MLOps.
    """

    def __init__(self, history_path="history.csv", mlflow_uri="http://127.0.0.1:5050"):
        # Initialisation des chemins
        self.history_path = os.path.abspath(history_path)
        self.mlflow_uri = mlflow_uri
        
        # Initialisation du lien MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialise le tracking MLflow vers le serveur local."""
        try:
            mlflow.set_tracking_uri("http://192.168.100.9:5050")
            mlflow.set_experiment("Agri_Intelligence_Production")
        except Exception as e:
            # On ne bloque pas l'UI si MLflow est éteint
            print(f"MLflow Offline: {e}")
            

    def log_inference(self, inputs: dict, verdict: str, confidence: str):
        """Enregistre les métriques en direct sur le serveur MLflow."""
        try:
            # Nettoyage du score (ex: '85.5%' -> 85.5)
            clean_score = float(str(confidence).replace('%', '').strip())
            
            with mlflow.start_run(run_name=f"Inference_{datetime.now().strftime('%H%M%S')}"):
                mlflow.log_params(inputs)
                mlflow.log_metric("confidence_score", clean_score)
                mlflow.set_tag("verdict", verdict)
                mlflow.set_tag("env", "production")
        except Exception as e:
            st.warning(f"MLflow Sync Warning: {e}")

    @st.cache_data(ttl=5) # Refresh ultra-rapide pour la démo
    def _fetch_monitored_data(_self):
        """Transforme le log d'audit local en DataFrame analytique."""
        if not os.path.exists(_self.history_path):
            return pd.DataFrame()

        try:
            # Lecture robuste du CSV
            df_raw = pd.read_csv(_self.history_path, header=None, on_bad_lines='skip').astype(str)
            refined = []
            status_keys = ["ELIGIBLE", "STANDARD", "APPROVED", "RISK", "REJECTED"]
            
            for _, row in df_raw.iterrows():
                vals = [v.strip() for v in row.tolist() if v.lower() != 'nan']
                v_key = next((v.upper() for v in vals if any(k in v.upper() for k in status_keys)), "UNKNOWN")
                c_val = next((v for v in vals if "%" in v or (v.replace('.','').isdigit() and 5 < float(v) <= 100)), "0%")
                t_val = next((v for v in vals if ":" in v), "Live")
                
                if v_key != "UNKNOWN":
                    refined.append({"Time": t_val, "Verdict": v_key, "Confidence (%)": c_val})
            
            return pd.DataFrame(refined)
        except Exception:
            return pd.DataFrame()

    def render_monitoring_ui(self):
        """Rendu visuel du Dashboard de Monitoring Professionnel."""
        df = self._fetch_monitored_data()

        if df.empty:
            st.info("🛰️ **System Standby:** En attente du flux de données d'analyse...")
            return

        # --- HEADER KPI ---
        st.markdown("### 📊 État de Santé du Système")
        k1, k2, k3, k4 = st.columns(4)
        
        scores = pd.to_numeric(df['Confidence (%)'].str.replace('%', ''), errors='coerce')
        avg_score = scores.mean()
        
        with k1:
            st.metric("Inférences Totales", len(df))
        with k2:
            st.metric("Confiance Moyenne", f"{avg_score:.1f}%", 
                      delta="Optimal" if avg_score > 75 else "Stable")
        with k3:
            approved = df[df['Verdict'].isin(['ELIGIBLE', 'STANDARD', 'APPROVED'])].shape[0]
            st.metric("Taux de Succès", f"{(approved/len(df))*100:.1f}%")
        with k4:
            st.metric("Moteur MLOps", "ONLINE", delta="PORT 5050", delta_color="normal")

        # --- VISUALISATIONS ---
        st.divider()
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.write("**📈 Tendance de Fiabilité (25 dernières)**")
            st.area_chart(scores.tail(25), width="stretch")

        with col_right:
            st.write("**⚖️ Distribution des Verdicts**")
            st.bar_chart(df['Verdict'].value_counts())

        # --- GOUVERNANCE ---
        with st.expander("🛡️ Audit de Gouvernance & Intégrité"):
            st.caption(f"Fichier source : `{self.history_path}`")
            st.dataframe(df.tail(10).iloc[::-1], width="stretch", hide_index=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger l'Audit complet", csv, "agri_audit.csv", "text/csv")
if 'monitor' not in st.session_state:
    st.session_state.monitor = SystemMonitor(history_path="history.csv")

monitor = st.session_state.monitor
# mlflow.log_tag

if not HAS_MLFLOW:
    st.sidebar.warning("⚠️ Mode Local : MLflow non installé")
else:
    st.sidebar.success("✅ MLflow Ready")

# --- 1. CONFIGURATION & PATHS ---
st.set_page_config(page_title="Agri-Resilience Command Center", page_icon="🌾", layout="wide")

# Chemins centralisés (Adaptés pour Docker & Disque E)
RAW_URL = os.environ.get("INFERENCE_API_URL", "http://agri_inference_api:8000")
INFERENCE_API_URL = RAW_URL.rstrip("/")  # Évite les doubles slashes //

PROJECT_ROOT = Path("/app")
DATA_STORAGE = PROJECT_ROOT / "data_storage"
MODEL_STORAGE = PROJECT_ROOT / "model_storage" 
DATASET_PATH = DATA_STORAGE / "raw" / "final_scoring.parquet"
HISTORY_PATH = DATA_STORAGE / "predictions" / "training_logs.csv"

# Style Premium
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .prediction-card { padding: 20px; border-radius: 15px; border-left: 5px solid #2e7d32; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (Optimisé) ---
@st.cache_data(show_spinner="Chargement du Master Data...")
def load_data():
    return pd.read_parquet(DATASET_PATH) if DATASET_PATH.exists() else pd.DataFrame()

def get_history_count():
    if not HISTORY_PATH.exists(): return 0
    try:
        return sum(1 for _ in open(HISTORY_PATH)) - 1
    except:
        return 0

def save_prediction(data, pred, confidence):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {**data, "target_label": pred, "confidence": confidence, "timestamp": datetime.now()}
    pd.DataFrame([record]).to_csv(HISTORY_PATH, mode='a', header=not HISTORY_PATH.exists(), index=False)

# Chargement initial
df = load_data()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2548/2548553.png", width=80)
    st.title("Control Tower")
    
    # Vérification Health Check Robuste
    api_ready = False
    try:
        # On interroge la route /health que nous avons créée dans l'API
        health_resp = requests.get(f"{INFERENCE_API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            api_ready = health_resp.json().get("status") == "ready"
    except:
        api_ready = False

    if api_ready:
        st.success("● API: ONLINE & READY")
    else:
        st.error("○ API: OFFLINE / LOADING")

    st.divider()

    if st.button("🎲 Random Scenario", width="stretch") and not df.empty:
        st.session_state.idx = np.random.randint(0, len(df))
        st.rerun()

    idx = st.number_input("Pointer", 0, max(0, len(df)-1), value=st.session_state.get('idx', 0))
    st.session_state.idx = idx

# --- 4. MAIN INTERFACE ---
if df.empty:
    st.warning("⚠️ Waiting for data in `data_storage/raw` on disk E...")
    st.stop()

# Séparation des colonnes techniques
feature_cols = [c for c in df.columns if c not in ['field_id', 'loan_status', 'index', 'resilience_score',"avg_yield","credit_score"]]

# css for nice 

st.markdown("""
<style>
/* Valeur principale */
div[data-testid="stMetricValue"] {
    color: black !important;
}

/* Label (le texte en haut) */
div[data-testid="stMetricLabel"] > div {
    color: black !important;
}

/* Delta (si tu en utilises) */
div[data-testid="stMetricDelta"] {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# KPI Banner
# --- 1. DATA PREPARATION ---
total_samples = f"{len(df):,}"
features_count = len(feature_cols)
main_val = df['resilience_score'].mean() if 'resilience_score' in df.columns else df[feature_cols[0]].mean()
resilience_val = f"{main_val:.2f}"
history_count = get_history_count()

# --- 2. LAYOUT & STYLE ---
c1, c2, c3, c4 = st.columns(4)

def render_kpi_card(column, label, value, icon="📊"):
    column.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 20px;
            text-align: left;
            transition: transform 0.3s ease;
        ">
            <p style="
                color: #808080; 
                font-size: 0.85rem; 
                margin-bottom: 5px; 
                text-transform: uppercase; 
                letter-spacing: 1px;
            ">{icon} {label}</p>
            <h2 style="
                color: var(--text-color); 
                margin: 0; 
                font-size: 1.8rem; 
                font-weight: 700;
                font-family: 'Inter', sans-serif;
            ">{value}</h2>
        </div>
    """, unsafe_allow_html=True)

# --- 3. RENDERING ---
render_kpi_card(c1, "Total Samples", total_samples, "📈")
render_kpi_card(c2, "Features", features_count, "🧬")
render_kpi_card(c3, "Avg Resilience", resilience_val, "🛡️")
render_kpi_card(c4, "Logged Data", history_count, "📁")

st.markdown("<br>", unsafe_allow_html=True)
# Zone d'Analyse
left, right = st.columns([1, 1.5])
raw_row = df.iloc[st.session_state.idx][feature_cols].to_dict()

with left:
    st.subheader("🧪 Input Scenario")
    with st.container(border=True):
        final_input = {}  # <--- IL EST ICI !
        grid = st.columns(2)
        for i, col in enumerate(feature_cols):
            with grid[i % 2]:
                final_input[col] = st.number_input(f"📍 {col}", value=float(raw_row[col]), format="%.4f")
with right:
    st.subheader("📈 Radar Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[final_input[c] for c in feature_cols], theta=feature_cols, fill='toself', name='Current', line_color='#2e7d32'))
    fig.add_trace(go.Scatterpolar(r=[df[c].mean() for c in feature_cols], theta=feature_cols, fill='toself', name='Avg', line_color='#ffa000', opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, df[feature_cols].max().max()])), height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
    fig.update_polars(radialaxis=dict(visible=True, range=[0, None])) # Le 'None' s'adapte auto

# 5. PREDICTION ENGINE
st.divider()
def get_advisory(data):
    alerts = []
    ph = data.get('ph_level', 7)
    if ph < 5.8:
        alerts.append({"topic": "Soil Acidification", "action": "Immediate Liming required.", "icon": "🧪"})
    
    rain = data.get('final_precipitation', 1000)
    if rain < 750:
        alerts.append({"topic": "Hydraulic Deficit", "action": "Switch to drought-resistant cultivars.", "icon": "💧"})
    
    return alerts

if st.button("🚀 EXECUTE INTELLIGENCE ANALYSIS", type="primary", width="stretch"):
    if not api_ready:
        st.error("L'API n'est pas encore prête. Vérifiez vos conteneurs Docker.")
    else:
        # Correction du Payload (On s'assure que les clés correspondent à l'API)
        api_payload = {
            "final_precipitation": final_input.get("final_precipitation", 0.0),
            "ph_level": final_input.get("ph_level", 0.0),
            "nitrogen_content": final_input.get("nitrogen_content", 0.0),
            "organic_matter": final_input.get("organic_matter", 0.0)
        }

        try:
            with st.spinner("🧠 AI analyzes agricultural constants..."):
                import time
                time.sleep(0.5)
                resp = requests.post(f"{INFERENCE_API_URL}/predict", json={"features": api_payload}, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    label = str(data.get('predicted_label', 'N/A')).upper()
                    conf_val = data.get('confidence_score', 0)
                    
                    # --- LOGIQUE DE COULEUR ET STYLE CHIC ---
                    is_risk = any(x in label for x in ["RISK", "REJECTED", "HIGH"])
                    color = "#d32f2f" if is_risk else "#2e7d32"
                    
                    # --- AFFICHAGE DU RÉSULTAT (L'utilisateur voit ça direct !) ---
                    if not is_risk:
                        st.balloons()
                    
                    res_l, res_r = st.columns([2, 1])
                    
                    with res_l:
                        # Carte de Verdict Premium
                        st.markdown(f"""
                            <div style="background-color: white; padding: 25px; border-radius: 15px; border-left: 10px solid {color}; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                                <h4 style="color: #555; margin: 0;">VERDICT EXPERT SYSTEM</h4>
                                <h1 style="color: {color}; margin: 10px 0;">{label}</h1>
                                <p style="color: #666;">{'⚠️ Warning: Low resilience factors.' if is_risk else '✅ optimal agricole zone.'}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with res_r:
                        st.markdown("""
                            <style>
                            /* Card design for recommendations */
                            .stAlert {
                                border-radius: 10px !important;
                                border: none !important;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                            }
                            /* Metric styling */
                            [data-testid="stMetricValue"] {
                                font-family: 'Inter', sans-serif;
                                font-weight: 800;
                                color: #1B5E20 !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                        st.metric("AI Confidence Score", f"{conf_val:.2%}")
                        st.progress(conf_val)
                        
                        if is_risk:
                            st.markdown("### 🛠️ Strategic Mitigation Plan")
                            st.caption("Custom corrective actions based on your soil and climate data.")

                            # Logic Engine for Precision Agriculture
                            def get_advisory(data):
                                alerts = []
                                
                                # 1. PH ANALYSIS (Chemical Stability)
                                ph = data.get('ph_level', 7)
                                if ph < 5.8:
                                    alerts.append({
                                        "topic": "Soil Acidification",
                                        "value": f"pH {ph:.2f}",
                                        "impact": "High (Nutrient Lock)",
                                        "action": "Immediate Liming (CaCO3) required to unlock Phosphorus uptake.",
                                        "icon": "🧪"
                                    })
                                
                                # 2. WATER RESILIENCE (Hydrological Risk)
                                rain = data.get('final_precipitation', 1000)
                                if rain < 750:
                                    alerts.append({
                                        "topic": "Hydraulic Deficit",
                                        "value": f"{rain:.0f} mm/year",
                                        "impact": "Critical (Yield Gap)",
                                        "action": "Switch to drought-resistant cultivars and implement mulching to preserve moisture.",
                                        "icon": "💧"
                                    })

                                # 3. NUTRIENT RATIO (Nitrogen Efficiency)
                                n_content = data.get('nitrogen_content', 50)
                                if n_content < 30:
                                    alerts.append({
                                        "topic": "Nitrogen Depletion",
                                        "value": f"{n_content:.1f} mg/kg",
                                        "impact": "Medium (Stunted Growth)",
                                        "action": "Leguminous intercropping (Soy/Beans) to naturally fix atmospheric Nitrogen.",
                                        "icon": "🌱"
                                    })
                                
                                return alerts

                            # Rendering the Expert Cards
                            for advice in get_advisory(final_input):
                            # Détermination de la couleur selon l'impact
                                status_color = "#E74C3C" if "Critical" in advice['impact'] or "High" in advice['impact'] else "#F1C40F"
                                
                                st.markdown(f"""
                                    <div style="
                                        border-left: 5px solid {status_color};
                                        background-color: rgba(255, 255, 255, 0.05);
                                        padding: 15px;
                                        border-radius: 0 10px 10px 0;
                                        margin-bottom: 10px;
                                        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                                    ">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <span style="font-size: 1.1em; font-weight: bold; color: white;">{advice['icon']} {advice['topic']}</span>
                                            <code style="color: {status_color};">{advice['value']}</code>
                                        </div>
                                        <div style="margin-top: 8px; font-size: 0.9em; color: #BDC3C7;">
                                            <strong>Impact:</strong> {advice['impact']}
                                        </div>
                                        <div style="
                                            margin-top: 10px;
                                            padding: 10px;
                                            background-color: rgba(0, 0, 0, 0.2);
                                            border-radius: 5px;
                                            font-size: 0.95em;
                                            color: #ECF0F1;
                                            border: 1px dashed rgba(255,255,255,0.1);
                                        ">
                                            <b>💡 Expert Action:</b> {advice['action']}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                                with st.container(border=True):
                                    st.markdown(f"**{advice['icon']} {advice['topic']}** | {advice['value']}")
                                    st.markdown(f"**Impact:** `{advice['impact']}`")
                                    st.info(advice['action'])
                                if is_risk:
                                    from datetime import datetime
                                    run_id = datetime.now().strftime("%H%M%S")

                                    st.markdown("---")
                                    
                                    # Préparation des données
                                    xai_data = {
                                        "Ph level": final_input.get('ph_level', 7),
                                        "Final precipitation": final_input.get('final_precipitation', 1000) / 10,
                                        "Nitrogen": final_input.get('nitrogen_content', 50)
                                    }
                                    
                                    df_xai = pd.DataFrame(list(xai_data.items()), columns=['Feature', 'Value'])
                                    
                                    fig_xai = px.bar(
                                        df_xai, 
                                        x='Value', 
                                        y='Feature', 
                                        orientation='h',
                                        title="<b>📊 XAI : RISK FACTOR ANALYSIS</b>",
                                        color='Value',
                                        # Dégradé du Rouge (Risque) au Vert (Optimal)
                                        color_continuous_scale=['#E74C3C', '#F1C40F', '#2ECC71']
                                    )
                                    
                                    fig_xai.update_layout(
                                        height=300, 
                                        margin=dict(t=50, b=20, l=0, r=0),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font=dict(color="white", size=12),
                                        showlegend=False,
                                        xaxis_title="Relative Impact Value",
                                        yaxis_title=""
                                    )
                
                                    # Utilise une clé fixe maintenant qu'il n'y a qu'un seul graphique
                                    st.plotly_chart(
                                        fig_xai,
                                        use_container_width=True,
                                        key=f"xai_chart_session_{random.randint(0, 999999)}"
                                    )

                                
                                    
                            st.warning("🚨 **Eligibility Status:** Profile requires technical upgrades before financing.")
                        else:
                            st.success("💎 **Optimal Agriculture Profile**")
                            
                    st.markdown("All parameters meet the 'Green Label' standards for automated credit approval.")
                    st.subheader("🔊 Audio Summary")

                    text_to_speak = ""
                    if is_risk:
                        st.markdown("### 🛠️ Strategic Mitigation Plan")
                        advice_list = get_advisory(final_input)
                        if advice_list:
                            for a in advice_list:
                                st.warning(f"{a['icon']} **{a['topic']}**: {a['action']}")
                            text_to_speak = f"Analysis complete. Status: {label}. Recommendations: " + " . ".join([f"{a['topic']}. {a['action']}" for a in advice_list])
                        else:
                            text_to_speak = f"Analysis complete. Status: {label}. Profile requires technical review."
                    else:
                        st.success("💎 Optimal Agriculture Profile")
                        text_to_speak = "Optimal profile detected. All systems green."

                    # LE BOUTON AUDIO (Bien indenté sous le 'if resp.status_code == 200')
                    st.markdown("---")
                    audio_placeholder = st.empty()
                    if st.button("🔊 Listen to AI Advisory", key="btn_audio_final"):
                        with st.spinner("Generating voice..."):
                            try:
                                # On s'assure que text_to_speak contient bien tes conseils
                                if not text_to_speak:
                                    text_to_speak = "Analysis complete. Please check the results on screen."
                                
                                tts = gTTS(text=text_to_speak, lang='en')
                                fp = io.BytesIO()
                                tts.write_to_fp(fp)
                                fp.seek(0)
                                audio_b64 = base64.b64encode(fp.read()).decode()
                                
                                # On injecte l'audio directement dans le placeholder
                                audio_html = f"""
                                    <div style="background: #262730; padding: 20px; border-radius: 10px; border: 1px solid #444;">
                                        <p style="color: white; margin-bottom: 10px;">▶️ Playing AI Summary...</p>
                                        <audio controls autoplay style="width: 100%;">
                                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                                        </audio>
                                    </div>
                                """
                                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"TTS Error: {e}")

        except Exception as e:
            st.error(f"Connection failed: {e}")

# 6. HISTORIQUE
with st.expander("📊 Production Feedback Loop", expanded=True):
    if HISTORY_PATH.exists():
        try:
            df_raw = pd.read_csv(HISTORY_PATH, header=None, on_bad_lines='skip').astype(str)
            
            if not df_raw.empty:
                refined_data = []
                for _, row in df_raw.iterrows():
                    values = row.tolist()
                    # Ta logique de recherche (Verdict, Score, Time)
                    keys = ["APPROVED", "REJECTED", "ELIGIBLE", "RISK", "STANDARD"]
                    verdict = next((v.strip() for v in values if any(k in v.upper() for k in keys)), None)
                    score = next((v.strip() for v in values if "%" in v or ("." in v and len(v) < 6)), "N/A")
                    time_val = next((v.strip() for v in values if ":" in v and len(v) > 5), "Recent")

                    if verdict:
                        refined_data.append({"Time": time_val, "Verdict": verdict, "Confidence": score})

                if refined_data:
                    final_df = pd.DataFrame(refined_data)
                    
                    # --- 1. LOGIQUE DE SCORING CENTRALISÉE  ---
                    try:
        # On extrait la toute dernière entrée du fichier (la plus précise)
                        latest_entry = refined_data[-1]
                        raw_score = latest_entry['Confidence'].replace('%', '')
                        score_to_display = float(raw_score)
                        verdict_to_display = latest_entry['Verdict'].upper()

                        # Normalisation automatique (0-10 -> 0-100)
                        if score_to_display <= 10:
                            score_to_display *= 10

                        # SYSTÈME DE DÉCISION FINANCIER
                        if "RISK" in verdict_to_display or "REJECTED" in verdict_to_display:
                            grade, color, label = "C", "#C62828", "HIGH RISK / MARGINAL"
                            score_to_display = min(score_to_display, 40)
                        else:
                            if score_to_display >= 80:
                                grade, color, label = "A", "#2E7D32", "EXCELLENT / ELIGIBLE"
                            elif score_to_display >= 60:
                                grade, color, label = "B", "#F9A825", "GOOD / ELIGIBLE"
                            else:
                                # Ton cas spécifique : Éligible mais score faible
                                grade, color, label = "C", "#FF8F00", "ELIGIBLE / CAUTION REQUIRED"

                        # --- AFFICHAGE DU BADGE ---
                        st.markdown(f"""
                            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 2px solid {color}; text-align: center; margin-bottom: 20px;">
                                <p style="margin: 0; color: #888; font-size: 11px; font-weight: bold;">LATEST LOGGED ASSESSMENT</p>
                                <div style="font-size: 42px; font-weight: bold; color: {color};">{grade}</div>
                                <p style="margin: 0; color: {color}; font-size: 14px; font-weight: bold;">{label}</p>
                                <p style="margin: 5px 0 0 0; color: #555; font-size: 10px;">Source: {latest_entry['Time']}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        # En cas d'erreur de lecture, on n'affiche rien pour ne pas polluer l'UI
                        pass
                    
                    # --- AFFICHAGE DU TABLEAU DE MONITORING (Image 3b5ae0.png) ---
                    st.dataframe(final_df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
                else:
                    st.warning("Data found but format is unrecognized.")
            else:
                st.info("Log file is empty.")
        except Exception as e:
            st.error(f"Display Error: {e}")
    else:
        st.info("Waiting for the first analysis...")

# --- 7. LIVE MONITORING RENDERING ---

st.sidebar.divider()
if st.sidebar.checkbox("📡 Live monitoring", value=True):
    st.write("---")
    st.subheader("📊 Intelligence System Monitoring")
    
    try:
        # ON LIT LE FICHIER QUE TU REMPLIS (training_logs.csv)
        # Assure-toi que HISTORY_PATH pointe bien sur training_logs.csv
        monitor_df = pd.read_csv(HISTORY_PATH) 
        
        if not monitor_df.empty:
            # --- DATA PREPARATION ---
            # 1. Ensure numeric confidence and scale to 100%
            monitor_df['conf_scale'] = pd.to_numeric(monitor_df['confidence'], errors='coerce')
            if monitor_df['conf_scale'].max() <= 1.0:
                monitor_df['conf_scale'] = monitor_df['conf_scale'] * 100
            
            # 2. Create the missing 'df_counts' for the summary graphs
            df_counts = monitor_df['target_label'].value_counts().reset_index()
            df_counts.columns = ['verdict', 'count']

            # --- DISPLAY GRID ---
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            with row1_col1:
                # DESIGN: Spline curve + Area fill pour la stabilité
                fig_rel = px.line(monitor_df.tail(20), y='conf_scale', 
                                 title="<b>SYSTEM RELIABILITY TREND</b>")
                
                fig_rel.update_traces(
                    line=dict(color='#00C853', width=4, shape='spline'), # Spline = Courbe fluide
                    fill='tozeroy', 
                    fillcolor='rgba(0, 200, 83, 0.1)',
                    mode='lines' # On enlève les markers pour un look plus "épuré"
                )
                
                # Seuil de confiance expert
                fig_rel.add_hrect(y0=0, y1=75, fillcolor="#FF5252", opacity=0.05, line_width=0)
                fig_rel.add_hline(y=75, line_dash="dot", line_color="#FF5252", annotation_text="Risk Zone")
                
                fig_rel.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', 
                                     yaxis_title="Confidence (%)", xaxis_title="Last Inferences")
                st.plotly_chart(fig_rel, use_container_width=True)

            with row2_col2:
                # DESIGN: Boxplot minimaliste (Focus sur les Outliers)
                fig_box = px.box(monitor_df, y='conf_scale', 
                                title="<b>PREDICTION VARIANCE</b>",
                                points="outliers", # Ne montre que les anomalies
                                color_discrete_sequence=['#FFC107'])
                
                fig_box.update_layout(height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig_box, use_container_width=True)
                
            with row1_col2:
                # 1. NETTOYAGE : On récupère uniquement le verdict (fin de la phrase)
                # On utilise 'Verdict' ou 'target_label' selon ce qui est dispo
                target_col = 'Verdict' if 'Verdict' in monitor_df.columns else 'target_label'
                
                # Cette ligne est la clé : elle ignore les dates et ne garde que le mot final
                monitor_df['clean_status'] = monitor_df[target_col].astype(str).apply(lambda x: x.split()[-1].upper())

                # 2. AGRÉGATION : On compte combien on a de chaque type
                df_stats = monitor_df['clean_status'].value_counts().reset_index()
                df_stats.columns = ['Decision', 'Total']

                # 3. LE GRAPHIQUE : Un vrai donut bien rond
                fig_donut = px.pie(
                    df_stats, 
                    values='Total', 
                    names='Decision', 
                    hole=0.7,
                    title="<b>STRATEGIC DECISION MIX</b>",
                    color='Decision',
                    color_discrete_map={
                        'APPROVED': '#2E7D32',
                        'STANDARD_ELIGIBLE': '#2E7D32',
                        'REJECTED': '#C62828',
                        'HIGH_RISK': '#F9A825'
                    }
                )

                # 4. ÉPURATION DU STYLE
                fig_donut.update_traces(
                    textinfo='percent+label', 
                    textposition='outside', # Sortir les textes pour que ce soit lisible
                    marker=dict(line=dict(color='#1E1E1E', width=2))
                )

                fig_donut.update_layout(
                    showlegend=False, 
                    height=450,
                    margin=dict(t=80, b=50, l=100, r=100) # Marges larges pour les étiquettes
                )

                st.plotly_chart(fig_donut, use_container_width=True)
                
            with row2_col1:
                # DESIGN: Histogramme de throughput pro
                fig_vol = px.bar(df_counts, x='verdict', y='count', 
                                title="<b>OPERATIONAL THROUGHPUT</b>",
                                text_auto='.2s') # Affiche les nombres (ex: 1.2k) sur les barres
                
                fig_vol.update_traces(marker_color='#455A64', marker_line_width=0)
                fig_vol.update_layout(height=300, xaxis_title="", yaxis_title="Total Logs")
                st.plotly_chart(fig_vol, use_container_width=True)

    except Exception as e:
        # Si le fichier n'existe pas encore au premier lancement
        st.warning(f"Le système de monitoring s'initialise... ({e})")