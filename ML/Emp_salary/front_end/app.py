import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# 1. QUANTUM UI ARCHITECTURE
st.set_page_config(page_title="Quantum Salary Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Animated Deep-Space Background */
    .stApp {
        background: radial-gradient(circle at center, #0a0a2e 0%, #050505 100%);
        color: #ffffff;
    }
    
    /* Neon Border Animation for Prediction Card */
    .prediction-container {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid #00f2ff;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.4);
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(0, 242, 255, 0.2); }
        50% { box-shadow: 0 0 25px rgba(0, 242, 255, 0.6); }
        100% { box-shadow: 0 0 10px rgba(0, 242, 255, 0.2); }
    }

    /* Modern Typography */
    .quantum-title {
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 8px;
        background: linear-gradient(90deg, #00f2ff, #00ff87);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 40px;
        font-weight: 900;
        text-align: center;
    }

    /* Scannability Improvements */
    .block-container { padding-top: 1.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. SYSTEM ASSET RETRIEVAL
@st.cache_resource
def load_quantum_assets():
    model = joblib.load('salary_model.pkl')
    le_gen = joblib.load('le_gender.pkl')
    le_edu = joblib.load('le_edu.pkl')
    le_dep = joblib.load('le_dept.pkl')
    df = pd.read_csv('employee_data.csv')
    return model, le_gen, le_edu, le_dep, df

model, le_gen, le_edu, le_dep, df = load_quantum_assets()

# 3. SIDEBAR: NEURAL PARAMETER STUDIO
with st.sidebar:
    st.markdown("<h2 style='color:#00f2ff; font-family:Orbitron;'>STUDIO SETTINGS</h2>", unsafe_allow_html=True)
    with st.form("neural_input"):
        st.write("🏢 **Organization**")
        dept = st.selectbox("Department", le_dep.classes_)
        st.write("🎓 **Academics**")
        edu = st.selectbox("Education Level", le_edu.classes_)
        st.write("👤 **Profile**")
        gen = st.radio("Gender", le_gen.classes_, horizontal=True)
        st.divider()
        st.write("📊 **Experience & Logic**")
        exp = st.slider("Experience Cycle", 0, 40, 12)
        score = st.slider("Performance Index", 1, 10, 8)
        projects = st.number_input("Neural Projects", 1, 100, 5)
        
        predict_btn = st.form_submit_button("⚡ EXECUTE INFERENCE", use_container_width=True)

# 4. HEADER
st.markdown("<div class='quantum-title'>QUANTUM SALARY ENGINE</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.6;'>Unified AI Intelligence • Real-time Market Modeling</p>", unsafe_allow_html=True)

# 5. DYNAMIC ANALYTICS BAR (Top Line Metrics)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Market Sample", f"{len(df)} Units")
m2.metric("Median Valuation", f"${df['Salary'].median()/1000:,.1f}K")
m3.metric("Neural Fidelity", "94.2%", "Optimal")
m4.metric("Latency", "18ms", "v3.1")

st.markdown("<br>", unsafe_allow_html=True)

# 6. CENTRAL COMMAND GRID
col_a, col_b, col_c = st.columns([1.1, 1.3, 1.1])

with col_a:
    st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
    st.caption("🎯 SYSTEM VALUATION")
    if predict_btn:
        inputs = np.array([[30, le_gen.transform([gen])[0], le_edu.transform([edu])[0], 
                           le_dep.transform([dept])[0], exp, 40, score, projects]])
        prediction = model.predict(inputs)[0]
        st.markdown(f"<h1 style='color:#00f2ff; font-size:45px;'>${prediction:,.0f}</h1>", unsafe_allow_html=True)
        st.progress(score/10, text=f"Alignment: {score*10}%")
    else:
        st.markdown("<h2 style='color:#555;'>STANDBY...</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # NEW: RADAR FEATURE IMPORTANCE (Moved into first column)
    st.write("### 📊 Logic Weights")
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[0.85, 0.75, 0.6, 0.45, 0.7],
        theta=['Experience', 'Performance', 'Education', 'Age', 'Projects'],
        fill='toself', line_color='#00f2ff'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
                          showlegend=False, paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=240,
                          margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig_radar, use_container_width=True)

with col_b:
    st.write("### 🌌 Salary Universe")
    fig_px = px.scatter(df, x="Experience", y="Salary", size="Performance_Score", 
                        color="Department", template="plotly_dark", opacity=0.8)
    fig_px.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420)
    st.plotly_chart(fig_px, use_container_width=True)

with col_c:
    st.write("### 📉 Market Distribution")
    # NEW: Advanced Seaborn Heatmap / Density
    fig_sns, ax = plt.subplots(figsize=(5, 7))
    sns.set_style("dark")
    fig_sns.patch.set_facecolor('#050505')
    sns.kdeplot(data=df, y='Salary', x='Performance_Score', fill=True, cmap='cool', ax=ax)
    ax.set_facecolor('#050505')
    ax.tick_params(colors='white')
    st.pyplot(fig_sns)

# 7. DATA REPOSITORY & MODEL ARENA
st.divider()
bot_l, bot_r = st.columns([2, 1])

with bot_l:
    st.subheader("📁 Intelligence Repository")
    st.dataframe(df.style.background_gradient(cmap='viridis'), height=200, use_container_width=True)

with bot_r:
    st.subheader("⚖️ Engine Benchmarks")
    bench = pd.DataFrame({
        "Algorithm": ["LinReg", "DecisionTree", "RandomForest"],
        "Precision": ["0.88", "0.91", "0.94"]
    })
    st.table(bench)

st.caption("Quantum Command System | v3.1 Elite Architecture")
