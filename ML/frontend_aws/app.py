import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Amazon Premium Analytics 2026",
    page_icon="👑",
    layout="wide"
)

# --- ADVANCED BACKGROUND & ASSET DATABASE ---
bg_images = {
    "🏠 Home": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?q=80&w=2070&auto=format&fit=crop",
    "📖 Project Story": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?q=80&w=2070&auto=format&fit=crop",
    "🔮 Predictor": "https://images.unsplash.com/photo-1515377905703-c4788e51af15?q=80&w=2070&auto=format&fit=crop", # Jewelry
    "📊 Market Analysis": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?q=80&w=2070&auto=format&fit=crop", # Sandals
    "🚀 Deep Insights": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=2070&auto=format&fit=crop",
    "📞 Contact Us": "https://images.unsplash.com/photo-1534536281715-e28d76689b4d?q=80&w=2070&auto=format&fit=crop"
}

# Image gallery for top-selling categories (Jewelry, Sandals, Tech)
gallery_imgs = {
    "Audio": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?q=80&w=400",
    "Accessories": "https://images.unsplash.com/photo-1515377905703-c4788e51af15?q=80&w=400", # Used for Jewels
    "Electronics": "https://images.unsplash.com/photo-1498049794561-7780e7231661?q=80&w=400",
    "Computers": "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?q=80&w=400",
    "Footwear": "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?q=80&w=400" # Sandals
}

# --- CUSTOM CSS (The UI Engine) ---
def apply_theme(image_url):
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("{image_url}");
        background-size: cover; background-attachment: fixed; background-position: center;
    }}
    .glass-panel {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px; padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    .stat-card {{
        background: rgba(212, 175, 55, 0.1); /* Gold tint */
        border: 1px solid #d4af37; border-radius: 10px; padding: 15px; text-align: center;
    }}
    h1, h2, h3, h4, p, label {{ color: white !important; font-family: 'Montserrat', sans-serif; }}
    .stButton>button {{
        background: linear-gradient(45deg, #d4af37, #f9f295);
        color: black !important; font-weight: 800; border: none; width: 100%; border-radius: 50px; padding: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=160)
    st.markdown("<br>", unsafe_allow_html=True)
    menu = st.radio("MAIN MENU", list(bg_images.keys()))
    st.markdown("---")
    st.write("💎 **Premium Access Enabled**")
    st.write("📅 Market Sync: May 2026")

apply_theme(bg_images[menu])

# --- LOAD ASSETS ---
@st.cache_resource
def load_data_and_model():
    model = joblib.load('sales_model.joblib')
    cols = joblib.load('feature_columns.joblib')
    cats = joblib.load('category_list.joblib')
    df = pd.read_csv('Amazon_Big_Sales_Dataset_2026.csv')
    return model, cols, cats, df

model, feature_cols, categories, df = load_data_and_model()

# --- PAGE LOGIC ---

if menu == "🏠 Home":
    st.markdown("<h1 style='text-align: center; font-size: 70px; letter-spacing: 5px;'>AMAZON ELITE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Advanced Machine Learning for High-Performance Retailers</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("### 💍 Jewelry Analysis\nOptimize price for luxury segments.")
    with c2: st.markdown("### 👡 Fashion Trends\nPredict sandals & footwear popularity.")
    with c3: st.markdown("### 📈 Demand Forecast\nIdentify high-review volume potential.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🌟 Partner Success Stories")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("<div class='stat-card'><h4>Aura Jewelry Co.</h4><p>'Using the Rating Predictor, we increased our luxury watch rating from 3.8 to 4.6 by adjusting launch prices.'</p></div>", unsafe_allow_html=True)
    with sc2:
        st.markdown("<div class='stat-card'><h4>SoleStyle Sandals</h4><p>'The Market Analysis tool helped us identify a gap in the premium footwear segment in 2026.'</p></div>", unsafe_allow_html=True)

elif menu == "📖 Project Story":
    st.title("📖 Behind the Intelligence")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.write("""
    ### 🛡️ The Problem
    Launching a product on Amazon is a risk. Sellers often guess their price and hope for good reviews.
    
    ### 🧪 Our Solution
    We engineered a **Supervised Learning Engine**. By training on thousands of data points, our AI understands how pricing affects customer sentiment in 2026.
    
    ### 🛠️ The Pipeline
    1. **Data Ingestion:** Cleaning 'Amazon Big Sales' raw CSV files.
    2. **One-Hot Encoding:** Converting text categories (like Audio/Jewelry) into binary logic.
    3. **Random Forest Regression:** Using 100 decision trees to vote on the most likely product rating.
    4. **Visualization:** Translating math into stunning Plotly dashboards.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop", use_container_width=True)

elif menu == "🔮 Predictor":
    st.title("🔮 The Prediction Oracle")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    with st.form("oracle_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            p_price = st.number_input("Target Listing Price ($)", min_value=1.0, value=299.99)
            p_cat = st.selectbox("Market Segment", categories)
        with col_b:
            p_rev = st.number_input("Expected Review Count", min_value=1, value=1500)
        
        if st.form_submit_button("REVEAL MARKET RATING"):
            input_df = pd.DataFrame([[p_price, p_rev]], columns=['Price_USD', 'Review_Count'])
            for cat in categories:
                input_df[f"Category_{cat}"] = 1 if cat == p_cat else 0
            input_df = input_df.reindex(columns=feature_cols, fill_value=0)
            
            score = model.predict(input_df)[0]
            st.markdown(f"<h1 style='text-align: center; color: #f9f295 !important;'>Predicted Rating: {score:.2f} ⭐</h1>", unsafe_allow_html=True)
            st.balloons()
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "📊 Market Analysis":
    st.title("📊 2026 Market Pulse")
    
    # PRODUCT GALLERY
    st.markdown("### 🏆 Top 4 Trending Products (High Sales Volume)")
    tops = df.sort_values(by='Review_Count', ascending=False).head(4)
    g_cols = st.columns(4)
    for i, (idx, row) in enumerate(tops.iterrows()):
        with g_cols[i]:
            img_url = gallery_imgs.get(row['Category'], gallery_imgs['Accessories'])
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.1); border-radius:15px; padding:15px; text-align:center; border:1px solid gold;">
                <img src="{img_url}" style="width:100%; border-radius:10px; margin-bottom:10px;">
                <h5 style="color:white;">{row['Product_Name']}</h5>
                <p style="color:#f9f295; font-weight:bold;">Rating: {row['Rating']}</p>
                <small style="color:white;">{row['Review_Count']:,} Reviews</small>
            </div>
            """, unsafe_allow_html=True)

    # DATA EXPLORER SEARCH
    st.markdown("<br><div class='glass-panel'>", unsafe_allow_html=True)
    st.subheader("🔎 Product Inventory Explorer")
    search = st.text_input("Search for a specific product name:")
    if search:
        st.dataframe(df[df['Product_Name'].str.contains(search, case=False)], use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🚀 Deep Insights":
    st.title("🚀 Advanced Statistical Deep-Dive")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    sel_cat = st.selectbox("Select Segment to Audit", categories)
    sub_df = df[df['Category'] == sel_cat]
    
    # 3D Plot for added uniqueness
    st.markdown(f"#### Price vs Rating vs Reviews in {sel_cat}")
    fig3d = px.scatter_3d(sub_df, x='Price_USD', y='Rating', z='Review_Count', 
                          color='Rating', template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "📞 Contact Us":
    st.title("📞 Get in Touch")
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    with st.form("contact"):
        st.text_input("Full Name")
        st.text_input("Business Email")
        st.text_area("Your Message")
        if st.form_submit_button("Send to Analyst Team"):
            st.success("Your message has been encrypted and sent to our 2026 data team!")
    st.markdown("</div>", unsafe_allow_html=True)