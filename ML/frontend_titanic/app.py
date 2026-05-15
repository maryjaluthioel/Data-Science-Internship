import streamlit as st
import joblib
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Titanic Survival Oracle", page_icon="🚢", layout="wide")

# --- BACKGROUND IMAGE DATABASE ---
bg_images = {
    "🏛️ Historical Context": "https://images.unsplash.com/photo-1569949381669-ecf31ae8e613?q=80&w=2070&auto=format&fit=crop", # Old ship/harbor
    "🔮 Survival Predictor": "https://images.unsplash.com/photo-1500077423678-25eead4cb523?q=80&w=2070&auto=format&fit=crop"  # Ship at night
}

# --- CUSTOM CSS (Nautical Design) ---
def apply_nautical_theme(image_url):
    st.markdown(f"""
    <style>
    /* Dynamic background with deep blue tint */
    .stApp {{
        background: linear-gradient(rgba(10, 25, 47, 0.8), rgba(10, 25, 47, 0.8)), 
                    url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Frosted glass container */
    .glass-container {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(10, 25, 47, 0.9) !important;
    }}

    /* Typography */
    h1, h2, h3, p, label {{
        color: #f1f1f1 !important;
        font-family: 'Times New Roman', serif;
    }}

    /* Button gradient: Deep Sea to Ice */
    .stButton>button {{
        background: linear-gradient(90deg, #1b3d4f, #3a7bd5);
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: 0.5s;
        height: 3em;
        width: 100%;
    }}
    .stButton>button:hover {{
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.6);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/White_Star_Line_logo.svg", width=120)
    st.markdown("### Voyage Menu")
    page = st.radio("Choose Page:", list(bg_images.keys()))
    st.markdown("---")
    st.write("RMS Titanic Manifest Analysis")

apply_nautical_theme(bg_images[page])

# --- ASSETS ---
@st.cache_resource
def load_titanic_model():
    # Ensure this file exists in your folder!
    return joblib.load('titanic_model.pkl')

# --- PAGE LOGIC ---

if page == "🏛️ Historical Context":
    st.markdown("<h1 style='text-align: center; font-size: 55px;'>The Unsinkable Legend</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("""
        ### April 15, 1912
        The RMS Titanic was a British passenger liner operated by the White Star Line. 
        It sank in the North Atlantic Ocean in the early morning hours after striking an iceberg.
        
        ### The Dataset
        Our AI model analyzes the passenger manifest to determine which factors—such as age, gender, 
        and socioeconomic class—influenced survival rates during the disaster.
        """)
    with col2:
        st.image("https://images.unsplash.com/photo-1569949381669-ecf31ae8e613?q=80&w=800", caption="The Titanic Departure", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔮 Survival Predictor":
    st.markdown("<h1 style='text-align: center;'>Predictive Manifest</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    model = load_titanic_model()
    
    with st.form("survival_form"):
        c1, c2 = st.columns(2)
        with c1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            sex = st.radio("Gender", ["male", "female"])
            age = st.slider("Age of Passenger", 1, 80, 25)
        with c2:
            sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
            parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
            fare = st.number_input("Ticket Fare (£)", 0.0, 512.0, 32.0)
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button("CALCULATE SURVIVAL PROBABILITY")

    if predict_btn:
        # Pre-processing
        sex_val = 0 if sex == "male" else 1
        emb_map = {"S": 0, "C": 1, "Q": 2}
        emb_val = emb_map.get(embarked, 0)

        data = pd.DataFrame(
            [[pclass, sex_val, age, sibsp, parch, fare, emb_val]],
            columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        )

        result = model.predict(data)
        
        if result[0] == 1:
            st.markdown("<h2 style='color: #00ffcc !important; text-align: center;'>✨ SURVIVOR</h2>", unsafe_allow_html=True)
            st.write("This passenger profile matches the criteria for those who safely reached a lifeboat.")
        else:
            st.markdown("<h2 style='color: #ff4b4b !important; text-align: center;'>💀 PERISHED</h2>", unsafe_allow_html=True)
            st.write("The model predicts this passenger would not have survived the sinking.")
    
    st.markdown("</div>", unsafe_allow_html=True)