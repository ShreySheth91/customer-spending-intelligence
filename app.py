import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import xgboost as xgb
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Spending Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1318 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #c8cdd8 !important; }
[data-testid="stSidebar"] .stRadio label { font-family: 'DM Mono', monospace; font-size: 13px; }

/* Main background */
[data-testid="stAppViewContainer"] { background: #0b0f14; }
[data-testid="stAppViewContainer"] > .main { background: #0b0f14; }

/* Text colors */
h1, h2, h3, h4 { color: #e8eaf0 !important; font-family: 'DM Sans', sans-serif; font-weight: 600; }
p, li, span, div { color: #c8cdd8; }
.stMarkdown p { color: #9aa0b0; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #111620;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #7a8499 !important; font-family: 'DM Mono', monospace; font-size: 11px; }
[data-testid="stMetricValue"] { color: #e8eaf0 !important; font-size: 28px !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
.stSlider { background: #111620 !important; border-color: rgba(255,255,255,0.1) !important; color: #e8eaf0 !important; }

/* Buttons */
.stButton > button {
    background: #1a4a8a !important;
    color: #e8eaf0 !important;
    border: 1px solid rgba(78,155,255,0.3) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 8px 24px !important;
    transition: all 0.2s;
}
.stButton > button:hover { background: #1d5aaa !important; border-color: rgba(78,155,255,0.6) !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* Tables */
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader { background: #111620 !important; border-radius: 8px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #111620; border-radius: 8px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #7a8499; font-family: 'DM Mono', monospace; font-size: 12px; }
.stTabs [aria-selected="true"] { background: #1a4a8a !important; color: #e8eaf0 !important; }

/* Info / warning boxes */
.stAlert { border-radius: 8px !important; }

/* Plot background */
.element-container { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────
FRAUD_FEATURES = [
    'log_amt', 'amt_z_customer',
    'hour', 'hour_sin', 'hour_cos', 'dow', 'month', 'is_weekend', 'is_night',
    'distance_km',
    'age', 'gender_enc', 'city_pop',
    'category_enc',
    'cluster',
]

CLV_FEATURES = [
    'recency', 'frequency', 'avg_txn_amt', 'max_txn_amt', 'std_txn_amt',
    'night_txn_pct', 'weekend_txn_pct', 'unique_merchants',
    'unique_states', 'unique_categories', 'avg_distance_km',
    'customer_age', 'city_pop', 'cluster', 'fraud_flag',
]

CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net',
    'grocery_pos', 'health_fitness', 'home', 'kids_pets',
    'misc_net', 'misc_pos', 'personal_care', 'shopping_net',
    'shopping_pos', 'travel',
]

SEGMENT_COLORS = {
    'Champions':  '#4e9bff',
    'Loyal':      '#52d48b',
    'Suspicious': '#f05a5a',
    'Dormant':    '#f5a623',
}

SEGMENT_DESCRIPTIONS = {
    'Champions': {
        'desc': 'Most recent activity, highest frequency and spend. Your top-value customers.',
        'action': 'Prioritise retention. Reward loyalty, offer premium features.',
        'n': 396, 'recency': 0.03, 'frequency': 2062, 'monetary': 143687, 'fraud_rate': 75.0,
    },
    'Loyal': {
        'desc': 'Recent, high-frequency, solid lifetime spend (~$66K). Core revenue base.',
        'action': 'Maintain engagement. Monitor for recency drift — trigger re-engagement early.',
        'n': 512, 'recency': 0.26, 'frequency': 936, 'monetary': 66171, 'fraud_rate': 76.0,
    },
    'Suspicious': {
        'desc': 'Dormant 250+ days, very low frequency but anomalously high avg transaction ($640). Weekend-heavy (79%). 100% fraud-flagged.',
        'action': 'Immediate fraud review. Large infrequent weekend transactions on inactive accounts.',
        'n': 34, 'recency': 250.4, 'frequency': 10, 'monetary': 6333, 'fraud_rate': 100.0,
    },
    'Dormant': {
        'desc': 'Inactive ~300 days, minimal transactions. Older demographic (avg age 61). 100% fraud-flagged.',
        'action': 'Freeze accounts pending fraud review. Win-back unlikely given inactivity duration.',
        'n': 41, 'recency': 296.5, 'frequency': 10, 'monetary': 5539, 'fraud_rate': 100.0,
    },
}

plt.style.use('dark_background')
PLOT_BG    = '#111620'
PLOT_FG    = '#1e2535'
TEXT_COLOR = '#9aa0b0'
GRID_COLOR = 'rgba(255,255,255,0.05)'

def dark_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color('#e8eaf0')
    for spine in ax.spines.values():
        spine.set_edgecolor('rgba(255,255,255,0.08)')
    ax.grid(color='#1e2535', linewidth=0.5)
    return fig, ax

def dark_figs(nrows, ncols, w=14, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(PLOT_BG)
    axlist = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axlist:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color('#e8eaf0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2535')
        ax.grid(color='#1e2535', linewidth=0.5)
    return fig, axes

# ─────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    fraud_model = xgb.XGBClassifier()
    fraud_model.load_model("fraud_model.json")

    clv_model = xgb.XGBRegressor()
    clv_model.load_model("clv_model.json")

    with open("model_metadata.json") as f:
        meta = json.load(f)

    fraud_explainer = shap.TreeExplainer(fraud_model)
    clv_explainer   = shap.TreeExplainer(clv_model)

    return fraud_model, clv_model, meta, fraud_explainer, clv_explainer

@st.cache_data(show_spinner="Loading customer data...")
def load_segments():
    return pd.read_csv("customer_segments.csv")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px;'>
        <div style='font-family: DM Mono, monospace; font-size: 10px; color: #4e9bff;
                    letter-spacing: .14em; text-transform: uppercase; margin-bottom: 6px;'>
            // fintech portfolio
        </div>
        <div style='font-size: 20px; font-weight: 600; color: #e8eaf0; line-height: 1.2;'>
            Customer Spending<br>Intelligence
        </div>
        <div style='font-family: DM Mono, monospace; font-size: 11px; color: #5a6478;
                    margin-top: 8px;'>Shrey Sheth · 2024–25</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊  EDA Overview", "🔍  Fraud Scorer", "👥  Segment Explorer", "💰  CLV Estimator"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 10px; color: #3a4255;
                line-height: 1.9; padding-top: 4px;'>
        Dataset · 1.30M transactions<br>
        Customers · 983 accounts<br>
        Fraud rate · 0.58%<br>
        Phases · EDA → RFM → Model → App
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 1 — EDA OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "📊  EDA Overview":
    st.markdown("## 📊 EDA Overview")
    st.markdown("<p style='color:#7a8499;font-family:DM Mono,monospace;font-size:12px;'>Phase 1 · Credit Card Transactions Dataset · 1.30M rows · Jan 2019 – Jun 2020</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transactions", "1.30M")
    c2.metric("Total Spend",        "$91M")
    c3.metric("Unique Customers",   "983")
    c4.metric("Fraud Rate",         "0.58%",  delta="-99.42% legit", delta_color="inverse")
    c5.metric("Median Txn (legit)", "$47.28", delta="$396.50 fraud")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Fraud Distribution", "Category Analysis", "Temporal Patterns"])

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            fig, ax = dark_fig(6, 4)
            classes = ['Legitimate\n1,289,169', 'Fraudulent\n7,506']
            vals    = [1289169, 7506]
            colors  = ['#52d48b', '#f05a5a']
            bars = ax.bar(classes, vals, color=colors, edgecolor='#0b0f14', linewidth=1.5, width=0.4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 8000,
                        f'{v:,}', ha='center', fontsize=10, color='#e8eaf0', fontweight='500')
            ax.set_title('Transaction count by class', fontsize=12, fontweight='bold', pad=12)
            ax.set_ylabel('Count')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = dark_fig(6, 4)
            ax.set_facecolor(PLOT_BG)
            wedges, texts, autotexts = ax.pie(
                [1289169, 7506],
                labels=['Legitimate', 'Fraudulent'],
                autopct='%1.2f%%',
                colors=['#52d48b', '#f05a5a'],
                startangle=90,
                wedgeprops={'edgecolor': '#0b0f14', 'linewidth': 2},
                explode=[0, 0.08],
                textprops={'color': TEXT_COLOR, 'fontsize': 10},
            )
            for at in autotexts: at.set_color('#e8eaf0')
            ax.set_title('Class imbalance', fontsize=12, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.info("⚠️  **Severe class imbalance.** 0.58% fraud rate means accuracy is a useless metric — a naïve 'always predict legit' model scores 99.4%. Primary metric for the fraud model is **Precision-Recall AUC**.")

    with tab2:
        cat_data = {
            'grocery_pos':   {'spend': 14.46, 'fraud': 1.41},
            'shopping_pos':  {'spend':  9.31, 'fraud': 0.72},
            'shopping_net':  {'spend':  8.63, 'fraud': 1.76},
            'gas_transport': {'spend':  8.35, 'fraud': 0.47},
            'home':          {'spend':  7.17, 'fraud': 0.16},
            'kids_pets':     {'spend':  6.50, 'fraud': 0.21},
            'entertainment': {'spend':  6.04, 'fraud': 0.25},
            'misc_net':      {'spend':  5.12, 'fraud': 1.45},
            'misc_pos':      {'spend':  5.01, 'fraud': 0.31},
            'food_dining':   {'spend':  4.67, 'fraud': 0.17},
            'health_fitness':{'spend':  4.65, 'fraud': 0.15},
            'travel':        {'spend':  4.52, 'fraud': 0.29},
            'personal_care': {'spend':  4.35, 'fraud': 0.24},
            'grocery_net':   {'spend':  2.44, 'fraud': 0.53},
        }
        cats   = list(cat_data.keys())
        spends = [cat_data[c]['spend'] for c in cats]
        frauds = [cat_data[c]['fraud'] for c in cats]

        fig, axes = dark_figs(1, 2, w=14, h=6)

        sorted_spend = sorted(zip(spends, cats), reverse=True)
        axes[0].barh([c for _, c in sorted_spend],
                     [s for s, _ in sorted_spend],
                     color='#4e9bff', alpha=0.8, edgecolor='#0b0f14')
        axes[0].set_title('Total spend by category ($M)', fontweight='bold')
        axes[0].set_xlabel('Spend ($M)')
        axes[0].invert_yaxis()

        fraud_colors = ['#f05a5a' if f > 0.58 else '#52d48b' for f in frauds]
        sorted_fraud = sorted(zip(frauds, cats, fraud_colors))
        axes[1].barh([c for _, c, _ in sorted_fraud],
                     [f for f, _, _ in sorted_fraud],
                     color=[col for _, _, col in sorted_fraud], alpha=0.8, edgecolor='#0b0f14')
        axes[1].axvline(0.58, color='#f5a623', linestyle='--', linewidth=1.5, label='Avg 0.58%')
        axes[1].set_title('Fraud rate by category (%)', fontweight='bold')
        axes[1].set_xlabel('Fraud rate (%)')
        axes[1].legend(fontsize=9)
        axes[1].invert_yaxis()

        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("🔴 Red = above-average fraud rate · 🟢 Green = below-average · Online categories (shopping_net, misc_net) carry 2–3× the risk of physical POS")

    with tab3:
        hour_fraud = [2.1, 2.5, 2.8, 2.4, 1.9, 1.2, 0.6, 0.5, 0.4, 0.4, 0.45, 0.5,
                      0.5, 0.5, 0.5, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0, 1.4, 1.8]
        hour_vol   = [18000,14000,11000,9000,10000,18000,38000,62000,78000,82000,84000,85000,
                      84000,83000,82000,80000,79000,76000,72000,68000,60000,52000,40000,28000]

        fig, ax1 = dark_fig(13, 4)
        ax2 = ax1.twinx()

        ax1.bar(range(24), hour_vol, color='#4e9bff', alpha=0.5, label='Volume')
        ax1.set_xlabel('Hour of day'); ax1.set_ylabel('Transaction count', color='#4e9bff')
        ax1.tick_params(axis='y', labelcolor='#4e9bff')
        ax1.set_xticks(range(24))
        ax1.set_facecolor(PLOT_BG)

        ax2.plot(range(24), hour_fraud, color='#f05a5a', marker='o',
                 linewidth=2.5, markersize=5, label='Fraud rate %')
        ax2.set_ylabel('Fraud rate (%)', color='#f05a5a')
        ax2.tick_params(axis='y', labelcolor='#f05a5a')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax1.set_title('Transaction volume and fraud rate by hour of day', fontweight='bold')

        for spine in ax2.spines.values():
            spine.set_edgecolor('#1e2535')
        ax2.grid(False)

        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("🌙 Fraud spikes sharply at 12am–4am when cardholder volume is lowest — a strong fraud signal captured as `is_night` in the model")

# ─────────────────────────────────────────────────────────────
# PAGE 2 — FRAUD SCORER
# ─────────────────────────────────────────────────────────────
elif page == "🔍  Fraud Scorer":
    st.markdown("## 🔍 Fraud Scorer")
    st.markdown("<p style='color:#7a8499;font-family:DM Mono,monospace;font-size:12px;'>Phase 3 · XGBoost + SMOTE · Primary metric: PR-AUC · Enter a transaction to score it</p>", unsafe_allow_html=True)
    st.markdown("---")

    try:
        fraud_model, clv_model, meta, fraud_explainer, clv_explainer = load_models()
        threshold = meta.get('fraud_threshold', 0.25)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### Transaction details")

            amt = st.number_input("Transaction amount ($)", min_value=1.0, max_value=30000.0,
                                  value=250.0, step=10.0)
            hour = st.slider("Hour of transaction (0–23)", 0, 23, 14)
            dow  = st.selectbox("Day of week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            dow_enc = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'].index(dow)
            month = st.slider("Month (1–12)", 1, 12, 6)
            category = st.selectbox("Merchant category", CATEGORIES)
            cat_enc  = CATEGORIES.index(category)
            distance = st.number_input("Customer–merchant distance (km)", 0.0, 2000.0, 45.0, 5.0)

            st.markdown("#### Customer profile")
            age      = st.slider("Customer age", 18, 90, 42)
            gender   = st.radio("Gender", ["Female", "Male"], horizontal=True)
            gender_enc = 1 if gender == "Male" else 0
            city_pop = st.number_input("City population", 1000, 5000000, 150000, 10000)
            segment  = st.selectbox("Customer segment (from Phase 2)",
                                    ['Champions', 'Loyal', 'Suspicious', 'Dormant'])
            cluster_enc = ['Champions','Loyal','Suspicious','Dormant'].index(segment)
            cust_mean = st.number_input("Customer's usual avg transaction ($)", 1.0, 5000.0, 65.0, 5.0)

        with col2:
            st.markdown("#### Score")

            if st.button("Run fraud score →", use_container_width=True):
                # Feature engineering
                log_amt     = np.log1p(amt)
                amt_z       = (amt - cust_mean) / max(cust_mean * 0.5, 1)
                hour_sin    = np.sin(2 * np.pi * hour / 24)
                hour_cos    = np.cos(2 * np.pi * hour / 24)
                is_weekend  = 1 if dow_enc >= 5 else 0
                is_night    = 1 if hour <= 5 else 0

                X_input = np.array([[
                    log_amt, amt_z, hour, hour_sin, hour_cos,
                    dow_enc, month, is_weekend, is_night,
                    distance, age, gender_enc, city_pop,
                    cat_enc, cluster_enc
                ]])

                prob   = fraud_model.predict_proba(X_input)[0, 1]
                flagged = prob >= threshold

                # Score display
                color = '#f05a5a' if flagged else '#52d48b'
                label = '🚨 FLAGGED' if flagged else '✅ CLEAR'

                st.markdown(f"""
                <div style='background:#111620; border:1px solid {color}40;
                            border-left: 4px solid {color}; border-radius:10px;
                            padding:24px; margin-bottom:16px; text-align:center;'>
                    <div style='font-family:DM Mono,monospace; font-size:11px;
                                color:#7a8499; letter-spacing:.12em; text-transform:uppercase;
                                margin-bottom:8px;'>fraud probability</div>
                    <div style='font-size:52px; font-weight:700; color:{color};
                                line-height:1; margin-bottom:8px;'>{prob*100:.1f}%</div>
                    <div style='font-size:16px; color:{color}; font-weight:500;'>{label}</div>
                    <div style='font-family:DM Mono,monospace; font-size:10px;
                                color:#5a6478; margin-top:10px;'>threshold = {threshold:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

                # Key signals
                signals = []
                if is_night:           signals.append(("🌙 Late-night transaction", "high risk"))
                if amt_z > 2:          signals.append((f"💸 Amount {amt_z:.1f}σ above norm", "high risk"))
                if is_weekend:         signals.append(("📅 Weekend transaction", "moderate"))
                if distance > 200:     signals.append((f"📍 Distance {distance:.0f}km", "moderate"))
                if segment in ['Suspicious','Dormant']: signals.append((f"👤 Segment: {segment}", "high risk"))

                if signals:
                    st.markdown("**Risk signals detected:**")
                    for sig, level in signals:
                        color_s = '#f05a5a' if level == 'high risk' else '#f5a623'
                        st.markdown(f"<span style='color:{color_s}; font-size:13px;'>• {sig}</span>", unsafe_allow_html=True)

                # SHAP waterfall
                st.markdown("---")
                st.markdown("**Why this score? — SHAP explanation**")

                X_named  = pd.DataFrame(X_input, columns=FRAUD_FEATURES)
                shap_exp = fraud_explainer(X_named)

                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor(PLOT_BG)
                shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
                plt.title('Feature contributions to this fraud score', fontsize=11,
                          color='#e8eaf0', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            else:
                st.markdown("""
                <div style='background:#111620; border:1px solid rgba(255,255,255,0.07);
                            border-radius:10px; padding:32px; text-align:center; margin-top:16px;'>
                    <div style='font-size:36px; margin-bottom:12px;'>💳</div>
                    <div style='color:#5a6478; font-family:DM Mono,monospace; font-size:12px;'>
                        Fill in the transaction details<br>and click "Run fraud score →"
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"""
            <div style='background:#0f1318; border-radius:8px; padding:14px 16px;
                        font-family:DM Mono,monospace; font-size:11px; color:#5a6478; line-height:1.8;'>
                Model · XGBoost + SMOTE · Tuned via RandomizedSearchCV (30 iter)<br>
                Primary metric · PR-AUC = {meta.get('fraud_pr_auc', 0):.4f}<br>
                ROC-AUC · {meta.get('fraud_roc_auc', 0):.4f}<br>
                Threshold · {threshold:.3f} (optimal F1)
            </div>
            """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Model files not found. Make sure `fraud_model.json`, `clv_model.json`, and `model_metadata.json` are in the app directory.")

# ─────────────────────────────────────────────────────────────
# PAGE 3 — SEGMENT EXPLORER
# ─────────────────────────────────────────────────────────────
elif page == "👥  Segment Explorer":
    st.markdown("## 👥 Segment Explorer")
    st.markdown("<p style='color:#7a8499;font-family:DM Mono,monospace;font-size:12px;'>Phase 2 · K-Means clustering (k=4) on RFM + behavioural features · 983 customers</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Segment summary cards
    cols = st.columns(4)
    for col, (seg, info) in zip(cols, SEGMENT_DESCRIPTIONS.items()):
        color = SEGMENT_COLORS[seg]
        col.markdown(f"""
        <div style='background:#111620; border:1px solid {color}30;
                    border-top: 3px solid {color}; border-radius:10px;
                    padding:18px; height:100%;'>
            <div style='font-family:DM Mono,monospace; font-size:10px;
                        color:{color}; letter-spacing:.1em; text-transform:uppercase;
                        margin-bottom:6px;'>{seg}</div>
            <div style='font-size:28px; font-weight:600; color:#e8eaf0;
                        margin-bottom:4px;'>{info["n"]}</div>
            <div style='font-family:DM Mono,monospace; font-size:10px; color:#5a6478;'>
                customers · {info["n"]/983*100:.1f}%
            </div>
            <div style='font-size:12px; color:#7a8499; margin-top:10px; line-height:1.5;'>
                {info["desc"][:90]}...
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Segment detail
    selected = st.selectbox("Deep-dive into segment:", list(SEGMENT_DESCRIPTIONS.keys()))
    info  = SEGMENT_DESCRIPTIONS[selected]
    color = SEGMENT_COLORS[selected]

    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.markdown(f"""
        <div style='background:#111620; border:1px solid {color}30;
                    border-left:4px solid {color}; border-radius:8px; padding:20px;'>
            <div style='font-family:DM Mono,monospace; font-size:11px; color:{color};
                        letter-spacing:.1em; text-transform:uppercase; margin-bottom:12px;'>
                {selected}
            </div>
            <table style='width:100%; font-family:DM Mono,monospace; font-size:12px;'>
                <tr><td style='color:#5a6478; padding:4px 0;'>Customers</td>
                    <td style='color:#e8eaf0; text-align:right;'>{info["n"]} ({info["n"]/983*100:.1f}%)</td></tr>
                <tr><td style='color:#5a6478; padding:4px 0;'>Recency</td>
                    <td style='color:#e8eaf0; text-align:right;'>{info["recency"]} days</td></tr>
                <tr><td style='color:#5a6478; padding:4px 0;'>Frequency</td>
                    <td style='color:#e8eaf0; text-align:right;'>{info["frequency"]:,} txns avg</td></tr>
                <tr><td style='color:#5a6478; padding:4px 0;'>Monetary</td>
                    <td style='color:#e8eaf0; text-align:right;'>${info["monetary"]:,}</td></tr>
                <tr><td style='color:#5a6478; padding:4px 0;'>Fraud rate</td>
                    <td style='color:{"#f05a5a" if info["fraud_rate"] > 80 else "#f5a623"};
                               text-align:right;'>{info["fraud_rate"]}%</td></tr>
            </table>
            <div style='margin-top:16px; padding-top:14px;
                        border-top:1px solid rgba(255,255,255,0.06);'>
                <div style='font-size:11px; color:#5a6478; margin-bottom:6px;'>Recommended action</div>
                <div style='font-size:12px; color:#9aa0b0; line-height:1.6;'>{info["action"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Radar chart for selected segment
        labels   = ['Recency\n(recent)', 'Frequency', 'Monetary', 'Avg amount', 'Night txn %', 'Fraud rate']
        N        = len(labels)
        angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles  += angles[:1]

        # Normalised values per segment
        raw = {
            'Champions':  [1-0.03/300, 2062/2062, 143687/143687, 70/640, 0.18/0.31, 75/100],
            'Loyal':      [1-0.26/300, 936/2062,  66171/143687,  72/640, 0.23/0.31, 76/100],
            'Suspicious': [1-250/300,  10/2062,   6333/143687,   640/640,0.29/0.31, 100/100],
            'Dormant':    [1-296/300,  10/2062,   5539/143687,   560/640,0.31/0.31, 100/100],
        }

        fig = plt.figure(figsize=(7, 5))
        fig.patch.set_facecolor(PLOT_BG)
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor(PLOT_BG)

        for seg_name, vals in raw.items():
            v = vals + vals[:1]
            alpha = 0.85 if seg_name == selected else 0.15
            lw    = 2.5 if seg_name == selected else 0.8
            c     = SEGMENT_COLORS[seg_name]
            ax.plot(angles, v, color=c, linewidth=lw, alpha=alpha)
            ax.fill(angles, v, color=c, alpha=0.08 if seg_name == selected else 0.02)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['', '', ''], fontsize=6)
        ax.grid(color='#1e2535', linewidth=0.5)
        ax.spines['polar'].set_edgecolor('#1e2535')
        ax.tick_params(colors=TEXT_COLOR)
        ax.set_title(f'{selected} — radar profile', color='#e8eaf0',
                     fontsize=11, fontweight='bold', pad=15)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Try loading actual segment data for distribution charts
    try:
        seg_df = load_segments()
        if 'segment_name' in seg_df.columns:
            st.markdown("---")
            st.markdown("#### Feature distributions")
            feat = st.selectbox("Feature to plot:", ['monetary', 'frequency', 'recency',
                                                      'avg_txn_amt', 'night_txn_pct', 'unique_merchants'])
            fig, ax = dark_fig(12, 4)
            for seg_name in seg_df['segment_name'].unique():
                data = seg_df[seg_df['segment_name'] == seg_name][feat].dropna()
                cap  = data.quantile(0.99)
                data = data.clip(upper=cap)
                ax.hist(data, bins=30, alpha=0.55, label=seg_name,
                        color=SEGMENT_COLORS.get(seg_name, '#888'), edgecolor='none', density=True)
            ax.set_title(f'Distribution of {feat} by segment', fontweight='bold')
            ax.set_xlabel(feat.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# PAGE 4 — CLV ESTIMATOR
# ─────────────────────────────────────────────────────────────
elif page == "💰  CLV Estimator":
    st.markdown("## 💰 CLV Estimator")
    st.markdown("<p style='color:#7a8499;font-family:DM Mono,monospace;font-size:12px;'>Phase 3 · XGBoost regression · R²=0.997 · MAE=$1,586 · Predict customer lifetime value</p>", unsafe_allow_html=True)
    st.markdown("---")

    try:
        fraud_model, clv_model, meta, fraud_explainer, clv_explainer = load_models()

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### Customer profile")

            recency   = st.number_input("Recency (days since last transaction)", 0, 730, 30)
            frequency = st.number_input("Frequency (total transactions)", 1, 3000, 500)
            avg_txn   = st.number_input("Average transaction amount ($)", 1.0, 2000.0, 65.0, 5.0)
            max_txn   = st.number_input("Largest single transaction ($)", 1.0, 30000.0, 500.0, 50.0)
            std_txn   = st.number_input("Std dev of transaction amounts ($)", 0.0, 2000.0, 80.0, 10.0)

            st.markdown("#### Behavioural features")
            night_pct   = st.slider("Night transaction % (12am–6am)", 0.0, 1.0, 0.05, 0.01)
            weekend_pct = st.slider("Weekend transaction %", 0.0, 1.0, 0.35, 0.01)
            u_merchants = st.number_input("Unique merchants visited", 1, 700, 200)
            u_states    = st.number_input("Unique states transacted in", 1, 51, 3)
            u_cats      = st.number_input("Unique categories used", 1, 14, 8)
            avg_dist    = st.number_input("Avg customer-merchant distance (km)", 0.0, 500.0, 78.0, 5.0)

            st.markdown("#### Demographics")
            cust_age  = st.slider("Customer age", 18, 90, 42)
            city_pop2 = st.number_input("City population", 1000, 5000000, 150000, 10000)
            seg2      = st.selectbox("Segment", ['Champions', 'Loyal', 'Suspicious', 'Dormant'])
            cluster2  = ['Champions', 'Loyal', 'Suspicious', 'Dormant'].index(seg2)
            fraud_fl  = st.radio("Fraud flag", [0, 1], horizontal=True,
                                  format_func=lambda x: "No fraud" if x == 0 else "Has fraud txn")

        with col2:
            st.markdown("#### Estimated CLV")

            if st.button("Estimate lifetime value →", use_container_width=True):
                X_clv_in = np.array([[
                    recency, frequency, avg_txn, max_txn, std_txn,
                    night_pct, weekend_pct, u_merchants,
                    u_states, u_cats, avg_dist,
                    cust_age, city_pop2, cluster2, fraud_fl
                ]])

                X_clv_named = pd.DataFrame(X_clv_in, columns=CLV_FEATURES)
                pred_log    = clv_model.predict(X_clv_named)[0]
                pred_clv    = np.expm1(pred_log)

                # Segment benchmark
                benchmarks = {'Champions': 143687, 'Loyal': 66171, 'Suspicious': 6333, 'Dormant': 5539}
                bench = benchmarks[seg2]
                delta = pred_clv - bench
                delta_str = f"+${delta:,.0f}" if delta >= 0 else f"-${abs(delta):,.0f}"

                color_clv = '#52d48b' if pred_clv > bench else '#f5a623'

                st.markdown(f"""
                <div style='background:#111620; border:1px solid {color_clv}40;
                            border-left:4px solid {color_clv}; border-radius:10px;
                            padding:28px; text-align:center; margin-bottom:16px;'>
                    <div style='font-family:DM Mono,monospace; font-size:10px;
                                color:#7a8499; letter-spacing:.12em; text-transform:uppercase;
                                margin-bottom:8px;'>predicted lifetime value</div>
                    <div style='font-size:52px; font-weight:700; color:{color_clv};
                                line-height:1; margin-bottom:6px;'>
                        ${pred_clv:,.0f}
                    </div>
                    <div style='font-family:DM Mono,monospace; font-size:11px; color:#5a6478;'>
                        vs {seg2} segment avg ${bench:,} &nbsp;·&nbsp; {delta_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Key drivers
                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Frequency", f"{frequency:,}", f"{'↑ high' if frequency > 936 else '↓ low'} vs avg")
                c_b.metric("Avg txn", f"${avg_txn:,.0f}")
                c_c.metric("Recency", f"{recency}d", f"{'recent' if recency < 30 else 'stale'}")

                # SHAP waterfall for CLV
                st.markdown("---")
                st.markdown("**What drives this estimate? — SHAP breakdown**")

                shap_clv = clv_explainer(X_clv_named)
                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor(PLOT_BG)
                shap.plots.waterfall(shap_clv[0], max_display=10, show=False)
                plt.title('Feature contributions to CLV estimate', fontsize=11,
                          color='#e8eaf0', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            else:
                st.markdown("""
                <div style='background:#111620; border:1px solid rgba(255,255,255,0.07);
                            border-radius:10px; padding:32px; text-align:center; margin-top:16px;'>
                    <div style='font-size:36px; margin-bottom:12px;'>💰</div>
                    <div style='color:#5a6478; font-family:DM Mono,monospace; font-size:12px;'>
                        Fill in the customer profile<br>and click "Estimate lifetime value →"
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"""
            <div style='background:#0f1318; border-radius:8px; padding:14px 16px;
                        font-family:DM Mono,monospace; font-size:11px; color:#5a6478; line-height:1.8;'>
                Model · XGBoost Regressor · Log-transformed target<br>
                R² · {meta.get('clv_r2', 0):.4f} &nbsp;·&nbsp; MAE · ${meta.get('clv_mae', 0):,.0f}<br>
                Top drivers · frequency → unique_merchants → avg_txn_amt → recency
            </div>
            """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Model files not found. Make sure all model files are in the app directory.")
