import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Page Config
st.set_page_config(page_title="IrisOS v2.0", page_icon="🍃", layout="wide")

# 2. Advanced CSS for a Modern Web Interface
st.markdown("""
    <style>
    /* Background with a more subtle, high-end botanical feel */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1464618663641-bbdd760ae84a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }

    /* Header Styling */
    .header-container {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 0 0 30px 30px;
        margin-bottom: 30px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    /* Smaller, rounded image styling */
    .flower-img {
        width: 250px;
        height: 250px;
        object-fit: cover;
        border-radius: 50%;
        border: 4px solid #4CAF50;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    }

    /* Hide Streamlit Header/Footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_model():
    with open('svm_model.pkl', 'rb') as f:
        m = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        s = pickle.load(f)
    return m, s

model, scaler = load_model()

# --- TOP HEADER ---
st.markdown("""
    <div class="header-container">
        <h1 style='margin:0; font-family: sans-serif; letter-spacing: 2px;'>BOTANICAL <span style='color:#4CAF50'>AI</span></h1>
        <p style='margin:0; color:#aaa;'>Automated Species Identification & Analysis</p>
    </div>
""", unsafe_allow_html=True)

# --- NAVIGATION / BUTTONS SECTION ---
# Placing "Buttons" (Inputs) directly below the title as requested
st.markdown("### 🎚️ Laboratory Controls")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    sl = st.slider('Sepal Length', 4.0, 8.0, 5.8)
with col_b2:
    sw = st.slider('Sepal Width', 2.0, 5.0, 3.0)
with col_b3:
    pl = st.slider('Petal Length', 1.0, 7.0, 4.3)
with col_b4:
    pw = st.slider('Petal Width', 0.1, 3.0, 1.3)

# Process Data
input_df = pd.DataFrame({'sepal length (cm)': [sl], 'sepal width (cm)': [sw], 
                        'petal length (cm)': [pl], 'petal width (cm)': [pw]})
scaled_data = scaler.transform(input_df)
pred_idx = model.predict(scaled_data)[0]
probs = model.predict_proba(scaled_data)[0]
target_names = ['Setosa', 'Versicolor', 'Virginica']
result = target_names[pred_idx]

# --- MAIN VIEWPORT: FLOWER & TABLE ---
st.divider()
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.markdown("### 📸 Specimen View")
    # Mapping images
    images = {
        "Setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "Versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "Virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
    # Using HTML for smaller size and circle shape
    st.markdown(f"""
        <div style="display: flex; justify-content: center; padding: 10px;">
            <img src="{images[result]}" class="flower-img">
        </div>
        <h2 style="text-align:center; color:#4CAF50;">{result}</h2>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📋 Analysis Report")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Current Input Metrics:")
    st.table(input_df)
    st.write(f"**Confidence Score:** {probs[pred_idx]*100:.2f}%")
    
    # Confidence Bar
    st.progress(float(probs[pred_idx]))
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER BAR CHART ---
with st.expander("Show Probability Distribution Details"):
    chart_data = pd.DataFrame({'Probability': probs}, index=target_names)
    st.bar_chart(chart_data)