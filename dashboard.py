import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from datetime import datetime
from drift_detector import analyze_feature_drift, get_overall_health, compute_psi
from database import log_drift_run, get_drift_history
from groq import Groq

# Set page layout to wide and title
st.set_page_config(page_title="DriftWatch MLOps", page_icon="👀", layout="wide")

# Inject Global Styles
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 3.5rem 2rem 2rem 2rem;}
body {background-color: #080b12;}

/* Metric cards */
.metric-card {
    background: #0f1623;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-label {
    font-size: 11px;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
}
.metric-green { color: #10b981; }
.metric-yellow { color: #f59e0b; }
.metric-red { color: #ef4444; }
.metric-blue { color: #3b82f6; }

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 500;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Health banner */
.banner-stable {
    background: #052e16;
    border: 1px solid #166534;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-monitor {
    background: #1c1007;
    border: 1px solid #92400e;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-highrisk {
    background: #1c0707;
    border: 1px solid #991b1b;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 4px;
}
.banner-sub {
    font-size: 13px;
    opacity: 0.7;
}

/* Status pills */
.pill-stable {
    background: #052e16;
    color: #10b981;
    border: 1px solid #166534;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
}
.pill-monitor {
    background: #1c1007;
    color: #f59e0b;
    border: 1px solid #92400e;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
}
.pill-highrisk {
    background: #1c0707;
    color: #ef4444;
    border: 1px solid #991b1b;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
}

/* Dividers */
.divider {
    border: none;
    border-top: 1px solid #1e2d45;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# HEADER SECTION
col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown("""
        <div style="display:flex; align-items:baseline; gap:10px;">
            <h1 style="margin:0; padding:0; color:white; font-size:32px;">DriftWatch</h1>
            <span style="background:#1e2d45; color:#94a3b8; padding:2px 8px; border-radius:12px; font-size:12px; font-family:monospace;">v1.0 · Production</span>
        </div>
    """, unsafe_allow_html=True)
with col_right:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<div style="text-align:right; color:#4a6080; margin-top:15px; font-family:monospace;">{current_time}</div>', unsafe_allow_html=True)
    
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Load baseline resources
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('baseline/model.pkl')
        baseline_stats = joblib.load('baseline/baseline_stats.pkl')
        baseline_preds = joblib.load('baseline/baseline_predictions.pkl')
        return model, baseline_stats, baseline_preds
    except Exception as e:
        return None, None, None

model, baseline_stats, baseline_preds = load_artifacts()

if model is None:
    st.error("Baseline artifacts not found. Please run `python train_model.py` first.")
    st.stop()

numeric_features = list(baseline_stats.keys())

# --- SIDEBAR REDESIGN ---
st.sidebar.markdown('<div class="section-header">CONTROLS</div>', unsafe_allow_html=True)

# Custom file uploader style logic wrapper
uploaded_file = st.sidebar.file_uploader("Upload Data CSV", type=['csv'], label_visibility="collapsed")
if uploaded_file:
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    st.sidebar.markdown(f"""
    <div style="background:#0f1623; border:1px solid #10b981; border-radius:8px; padding:12px; margin-bottom:12px;">
        <div style="color:#10b981; font-weight:600; display:flex; align-items:center; gap:8px;">
            <span>✅</span> Uploaded
        </div>
        <div style="color:#94a3b8; font-size:12px; margin-top:4px;">{uploaded_file.name} ({file_size_kb:.1f} KB)</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background:#080b12; border:1px dashed #1e2d45; border-radius:8px; padding:12px; margin-bottom:12px; text-align:center; color:#4a6080; font-size:13px;">
        Drop new data CSV here
    </div>
    """, unsafe_allow_html=True)

run_btn = st.sidebar.button("Run Drift Analysis", type="primary", width="stretch")

st.sidebar.markdown('<hr class="divider" style="margin:16px 0;">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="section-header">MODEL REGISTRY</div>', unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background:#0f1623; border:1px solid #1e2d45; border-radius:8px; padding:12px; margin-bottom:12px; font-family:monospace; font-size:12px;">
    <div style="color:white; font-weight:bold; margin-bottom:8px; display:flex; justify-content:space-between;">
        <span>telco-churn-rf</span> <span style="color:#3b82f6;">#prod-model</span>
    </div>
    <div style="color:#94a3b8; margin-bottom:4px;">Dataset <span style="float:right; color:white;">Telco Customer</span></div>
    <div style="color:#94a3b8; margin-bottom:4px;">Algorithm <span style="float:right; color:white;">Random Forest</span></div>
    <div style="color:#94a3b8; margin-bottom:4px;">Features <span style="float:right; color:white;">4 Numeric</span></div>
    <div style="color:#94a3b8; margin-top:8px; padding-top:8px; border-top:1px dashed #1e2d45;">Training Date <span style="float:right; color:white;">Baseline</span></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<hr class="divider" style="margin:16px 0;">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="section-header">SYSTEM STATUS</div>', unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size:13px;">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        <div style="width:8px; height:8px; background:#10b981; border-radius:50%;"></div>
        <span style="color:white;">System Operational</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        <div style="width:8px; height:8px; background:#10b981; border-radius:50%;"></div>
        <span style="color:white;">Database Connected</span>
    </div>
    <div style="color:#4a6080; font-size:11px; margin-top:12px;">
        Last analyzed: Just now
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN AREA ---
if not uploaded_file and not st.session_state.get('run_analysis', False):
    st.info("👈 Please upload a CSV file and click 'Run Drift Analysis' to begin.")
    
    # Just show history if exists in styled tables
    history = get_drift_history()
    if history:
        with st.expander("Historical Logs & Overviews", expanded=True):
            st.markdown('<div class="section-header">Previous Analysis Runs</div>', unsafe_allow_html=True)
            hist_df = pd.DataFrame(history)
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(hist_df[['timestamp', 'filename', 'model_health', 'prediction_drift_psi']], width="stretch")
            
            fig = px.line(hist_df, x='timestamp', y='prediction_drift_psi', markers=True, title="Prediction PSI Over Time")
            fig.update_layout(
                paper_bgcolor="#080b12", plot_bgcolor="#0f1623", font_color="#94a3b8",
                xaxis=dict(showgrid=True, gridcolor="#1e2d45"),
                yaxis=dict(showgrid=True, gridcolor="#1e2d45")
            )
            fig.add_hline(y=0.2, line_dash="dash", line_color="#ef4444")
            fig.add_hline(y=0.1, line_dash="dash", line_color="#f59e0b")
            st.plotly_chart(fig, width="stretch")

if run_btn or st.session_state.get('run_analysis', False):
    st.session_state['run_analysis'] = True
    
    if uploaded_file is None:
        st.warning("Please upload a file before running the analysis.")
        st.stop()
        
    # Read upload
    new_df = pd.read_csv(uploaded_file)

    required_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    missing = [col for col in required_columns if col not in new_df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {missing}. Please upload the Telco Customer Churn dataset.")
        st.stop()

    new_df['TotalCharges'] = pd.to_numeric(new_df['TotalCharges'], errors='coerce')
    new_df['TotalCharges'].fillna(new_df['TotalCharges'].median(), inplace=True)
    new_df = new_df.dropna()  # handle NaNs gracefully
    
    # Analyze drift feature by feature
    feature_results = []
    statuses = []
    
    for feature in numeric_features:
        if feature in new_df.columns:
            base_data = baseline_stats[feature]['values']
            new_data = new_df[feature].values
            res = analyze_feature_drift(base_data, new_data, feature)
            feature_results.append(res)
            statuses.append(res['status'])
            
    X_new = new_df.drop(columns=['Churn', 'customerID'], errors='ignore')
    new_preds = model.predict_proba(X_new)[:, 1]
    
    pred_psi = compute_psi(baseline_preds, new_preds)
    
    model_health = get_overall_health(statuses)
    if pred_psi > 0.2:
        model_health = "HIGH RISK"
    elif pred_psi > 0.1 and model_health == "STABLE":
        model_health = "MONITOR"
        
    log_drift_run(uploaded_file.name, model_health, pred_psi, feature_results)

    # --- RENDER RESULTS ---
    
    if model_health == "STABLE":
        st.markdown(f"""
        <div class="banner-stable">
            <div style="font-size:24px;">✅</div>
            <div>
                <div class="banner-title">STABLE</div>
                <div class="banner-sub">Model performing normally. Prediction PSI is {pred_psi:.4f}.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif model_health == "MONITOR":
        st.markdown(f"""
        <div class="banner-monitor">
            <div style="font-size:24px;">⚠️</div>
            <div>
                <div class="banner-title">MONITOR</div>
                <div class="banner-sub">Moderate drift detected. Review drifting features closely.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="banner-highrisk">
            <div style="font-size:24px;">⛔</div>
            <div>
                <div class="banner-title">HIGH RISK</div>
                <div class="banner-sub">Significant data drift detected. Retraining is strongly recommended.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    c1, c2, c3, c4 = st.columns(4)
    total_f = len(feature_results)
    stable_f = sum(1 for x in statuses if x == "STABLE")
    monitor_f = sum(1 for x in statuses if x == "MONITOR")
    risk_f = sum(1 for x in statuses if x == "HIGH RISK")
    
    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Features</div>
        <div class="metric-value metric-blue">{total_f}</div>
        <div style="font-size:11px; color:#4a6080; margin-top:8px;">Analyzed tracked columns</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Stable Features</div>
        <div class="metric-value metric-green">{stable_f}</div>
        <div style="font-size:11px; color:#4a6080; margin-top:8px;">No drift detected</div>
    </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Monitor Features</div>
        <div class="metric-value metric-yellow">{monitor_f}</div>
        <div style="font-size:11px; color:#4a6080; margin-top:8px;">Warning threshold crossed</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High Risk</div>
        <div class="metric-value metric-red">{risk_f}</div>
        <div style="font-size:11px; color:#4a6080; margin-top:8px;">Critical drift observed</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-header">Drift Metrics Table</div>', unsafe_allow_html=True)
    
    # Custom HTML Table
    table_html = """
    <table style="width:100%; border-collapse:collapse; color:white; font-size:13px; text-align:left;">
        <thead>
            <tr style="background:#0f1623; border-bottom:1px solid #1e2d45;">
                <th style="padding:12px;">Feature</th>
                <th style="padding:12px;">PSI Score</th>
                <th style="padding:12px;">KL Divergence</th>
                <th style="padding:12px;">KS Statistic</th>
                <th style="padding:12px;">KS P-Value</th>
                <th style="padding:12px;">Status</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, row in enumerate(feature_results):
        bg_color = "#080b12" if i % 2 == 0 else "#0d1520"
        
        status = row['status']
        if status == "STABLE":
            pill = '<span class="pill-stable">STABLE</span>'
        elif status == "MONITOR":
            pill = '<span class="pill-monitor">MONITOR</span>'
        else:
            pill = '<span class="pill-highrisk">HIGH RISK</span>'
            
        table_html += f"""<tr style="background:{bg_color}; border-bottom:1px solid #1e2d45;">
            <td style="padding:12px; font-weight:500;">{row['feature']}</td>
            <td style="padding:12px;">{row['psi']:.4f}</td>
            <td style="padding:12px;">{row['kl_divergence']:.4f}</td>
            <td style="padding:12px;">{row['ks_stat']:.4f}</td>
            <td style="padding:12px;">{row['ks_pvalue']:.4f}</td>
            <td style="padding:12px;">{pill}</td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.write("")
    c_chart1, c_chart2 = st.columns(2)
    
    res_df = pd.DataFrame(feature_results)
    color_map = {"STABLE": "#10b981", "MONITOR": "#f59e0b", "HIGH RISK": "#ef4444"}
    res_df['color'] = res_df['status'].map(color_map)
    
    with c_chart1:
        st.markdown('<div class="section-header">Feature PSI Distribution</div>', unsafe_allow_html=True)
        fig_psi = px.bar(res_df, x='psi', y='feature', orientation='h', color='status', color_discrete_map=color_map)
        fig_psi.update_layout(
            paper_bgcolor="#080b12", plot_bgcolor="#0f1623", font_color="#94a3b8",
            xaxis=dict(showgrid=True, gridcolor="#1e2d45", title=""),
            yaxis=dict(showgrid=False, title="", showline=False),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False
        )
        fig_psi.add_vline(x=0.2, line_dash="dash", line_color="#ef4444")
        fig_psi.add_vline(x=0.1, line_dash="dash", line_color="#f59e0b")
        st.plotly_chart(fig_psi, width="stretch")
        
    with c_chart2:
        st.markdown('<div class="section-header">KS Test P-Values</div>', unsafe_allow_html=True)
        fig_ks = px.bar(res_df, x='feature', y='ks_pvalue', color='status', color_discrete_map=color_map)
        fig_ks.update_layout(
            paper_bgcolor="#080b12", plot_bgcolor="#0f1623", font_color="#94a3b8",
            xaxis=dict(showgrid=False, title="", showline=False),
            yaxis=dict(showgrid=True, gridcolor="#1e2d45", title=""),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False
        )
        fig_ks.add_hline(y=0.05, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(fig_ks, width="stretch")
        
    st.markdown('<div class="section-header">Prediction Score Drift</div>', unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Histogram(x=baseline_preds, name='Baseline', marker_color='#3b82f6', opacity=0.6, histnorm='probability density'))
    fig_pred.add_trace(go.Histogram(x=new_preds, name='New Data', marker_color='#ef4444', opacity=0.6, histnorm='probability density'))
    fig_pred.update_layout(
        barmode='overlay', 
        paper_bgcolor="#080b12", plot_bgcolor="#0f1623", font_color="#94a3b8",
        xaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Prediction Probability"),
        yaxis=dict(showgrid=True, gridcolor="#1e2d45", title="Density"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_pred, width="stretch")
    
    with st.expander("🔍 Feature Deep Dive", expanded=False):
        sel_feature = st.selectbox("Select Feature to Compare", res_df['feature'].tolist())
        
        if sel_feature:
            base_f_data = baseline_stats[sel_feature]['values']
            new_f_data = new_df[sel_feature].values
            
            c_f1, c_f2 = st.columns([2, 1])
            with c_f1:
                fig_f = go.Figure()
                fig_f.add_trace(go.Histogram(x=base_f_data, name='Baseline', marker_color='#3b82f6', opacity=0.6, histnorm='probability density'))
                fig_f.add_trace(go.Histogram(x=new_f_data, name='New Data', marker_color='#ef4444', opacity=0.6, histnorm='probability density'))
                fig_f.update_layout(
                    barmode='overlay', 
                    paper_bgcolor="#080b12", plot_bgcolor="#0f1623", font_color="#94a3b8",
                    xaxis=dict(showgrid=True, gridcolor="#1e2d45"),
                    yaxis=dict(showgrid=True, gridcolor="#1e2d45"),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_f, width="stretch")
                
            with c_f2:
                st.markdown("""<div style="font-size:13px; color:#4a6080; margin-bottom:8px; text-transform:uppercase;">Statistical Summary</div>""", unsafe_allow_html=True)
                s_table = f"""<table style="width:100%; border:1px solid #1e2d45; border-collapse:collapse; color:white; font-size:12px;">
                    <tr style="background:#0f1623;"><td style="padding:8px; border-bottom:1px solid #1e2d45;">Metric</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">Baseline</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">New Data</td></tr>
                    <tr style="background:#080b12;"><td style="padding:8px; border-bottom:1px solid #1e2d45;">Mean</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">{np.mean(base_f_data):.2f}</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">{np.mean(new_f_data):.2f}</td></tr>
                    <tr style="background:#0d1520;"><td style="padding:8px; border-bottom:1px solid #1e2d45;">Std Dev</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">{np.std(base_f_data):.2f}</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">{np.std(new_f_data):.2f}</td></tr>
                    <tr style="background:#080b12;"><td style="padding:8px; border-bottom:1px solid #1e2d45;">Min</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">{np.min(base_f_data):.2f}</td><td style="padding:8px; border-bottom:1px solid #1e2d45;">{np.min(new_f_data):.2f}</td></tr>
                    <tr style="background:#0d1520;"><td style="padding:8px;">Max</td><td style="padding:8px;">{np.max(base_f_data):.2f}</td><td style="padding:8px;">{np.max(new_f_data):.2f}</td></tr>
                </table>"""
                st.markdown(s_table, unsafe_allow_html=True)

    with st.expander("📋 Historical Drift Logs", expanded=False):
        history = get_drift_history()
        hist_df = pd.DataFrame(history)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        # build simple table for history to use pills
        history_table = '<table style="width:100%; border-collapse:collapse; color:white; font-size:12px; text-align:left;">'
        history_table += '<tr style="background:#0f1623; border-bottom:1px solid #1e2d45;"><th style="padding:8px;">Time</th><th style="padding:8px;">File</th><th style="padding:8px;">Pred PSI</th><th style="padding:8px;">Health</th></tr>'
        for i, row in hist_df.iterrows():
            bg_color = "#080b12" if i % 2 == 0 else "#0d1520"
            status = row['model_health']
            if status == "STABLE":
                pill = '<span class="pill-stable">STABLE</span>'
            elif status == "MONITOR":
                pill = '<span class="pill-monitor">MONITOR</span>'
            else:
                pill = '<span class="pill-highrisk">HIGH RISK</span>'
            history_table += f'<tr style="background:{bg_color}; border-bottom:1px solid #1e2d45;"><td style="padding:8px;">{row["timestamp"]}</td><td style="padding:8px;">{row["filename"]}</td><td style="padding:8px;">{row["prediction_drift_psi"]:.4f}</td><td style="padding:8px;">{pill}</td></tr>'
        history_table += '</table>'
        st.markdown(history_table, unsafe_allow_html=True)

# --- AI DRIFT ANALYST CHATBOT ---
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
    <span style="font-size:24px;">🤖</span>
    <span style="font-size:16px; font-weight:600; color:white;">AI Drift Analyst</span>
</div>
<div style="font-size:12px; color:#4a6080; margin-bottom:16px;">Powered by Llama 3.3 · Groq</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.chat-container {
    background-color: #0f1623;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #1e2d45;
    margin-top: 1rem;
    margin-bottom: 2rem;
}
.stChatMessage { background-color: transparent !important; }
[data-testid="chatAvatarIcon-user"] { background-color: #3b82f6 !important; }
[data-testid="chatAvatarIcon-assistant"] { background-color: #10b981 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.warning("⚠️ GROQ_API_KEY environment variable is not set. Please set it to enable the AI Drift Analyst.")
else:
    client = Groq(api_key=api_key)
    
    context_health = model_health if 'model_health' in locals() else "No analysis run yet"
    context_features = ", ".join(numeric_features) if 'numeric_features' in locals() else "None"
    context_results = json.dumps(feature_results) if 'feature_results' in locals() else "None"
    context_psi = f"{pred_psi:.4f}" if 'pred_psi' in locals() else "N/A"
    
    system_prompt = f"""
    You are an ML monitoring analyst assistant embedded in a 
    drift detection dashboard. You help data scientists and 
    business users understand model drift results.
    
    Current Analysis Results:
    - Model Health Status: {context_health}
    - Features Analyzed: {context_features}
    - Feature Drift Results: {context_results}
    - Prediction Drift PSI: {context_psi}
    
    PSI Interpretation:
    - PSI < 0.1: No significant drift (STABLE)
    - PSI 0.1-0.2: Moderate drift (MONITOR)
    - PSI > 0.2: Significant drift (HIGH RISK)
    
    KS Test: p-value < 0.05 means statistically significant drift.
    
    Answer questions about the drift results clearly and concisely.
    Give actionable recommendations when asked.
    If no analysis has been run yet, tell the user to upload a CSV first.
    Keep answers under 150 words unless asked for detail.
    """
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Suggested Questions (styled pills)
    col1, col2, col3, col4 = st.columns(4)
    suggested_q = None
    if col1.button("Should I retrain my model?", width="stretch"):
        suggested_q = "Should I retrain my model?"
    if col2.button("Which feature drifted most?", width="stretch"):
        suggested_q = "Which feature is drifting most?"
    if col3.button("Explain the PSI score", width="stretch"):
        suggested_q = "Explain the PSI score"
    if col4.button("What action should I take?", width="stretch"):
        suggested_q = "What action should I take?"
        
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Ask about your model's drift...")
    
    prompt = suggested_q or user_input
    
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *[{"role": m["role"], "content": m["content"]} 
                              for m in st.session_state.chat_messages]
                        ],
                        max_tokens=300
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error communicating with Groq: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
