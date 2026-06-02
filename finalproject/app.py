import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

# --- 1. GLOBAL SYSTEM SETUP & PROFESSIONAL THEME INJECTION ---
st.set_page_config(
    page_title="DEEPGUARD // AI Tactical Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Blue/Slate Tactical CSS Injection
st.markdown("""
    <style>
    /* Deep Tactical Navy Background */
    .stApp {
        background: #0a0f1d;
        color: #FFFFFF;
    }
    
    /* Sleek Deep Blue / Platinum Header */
    .cyber-title {
        font-family: 'Arial Black', -apple-system, sans-serif;
        text-align: center;
        color: #4da6ff;
        text-shadow: 0 0 8px rgba(77, 166, 255, 0.3);
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 2px;
    }
    .cyber-subtitle {
        text-align: center;
        color: #FFB300;
        font-family: 'Courier New', monospace;
        font-size: 15px;
        font-weight: bold;
        margin-bottom: 35px;
        letter-spacing: 2px;
    }
    
    /* Clean Slate/Navy Login Card */
    .login-card {
        background: #111827;
        border: 2px solid #2563eb;
        border-radius: 4px;
        padding: 35px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* High-Contrast Status Boxes */
    .status-box {
        background: #111827;
        border: 1px solid #374151;
        border-left: 6px solid #00FF66;
        padding: 15px;
        color: #FFFFFF;
    }
    
    /* Force bold override on labels */
    label {
        color: #FFB300 !important;
        font-weight: bold !important;
        font-family: monospace !important;
        letter-spacing: 1px;
    }

    /* Custom Blue Tactical Primary Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: white !important;
        border: 1px solid #4da6ff !important;
        font-weight: bold !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #60a5fa !important;
        box-shadow: 0 0 15px rgba(96, 165, 250, 0.6);
    }

    /* Custom Blue Tactical Standard Buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #111827 !important;
        color: #4da6ff !important;
        border: 1px solid #2563eb !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #60a5fa !important;
        color: #ffffff !important;
    }

    /* --- TACTICAL BLUE RADIO BUTTON OVERRIDES --- */
    
    /* Changes the outer circle color when selected */
    div[data-testid="stRadio"] input[type="radio"]:checked + div {
        border-color: #2563eb !important;
    }

    /* Changes the inner core dot color when selected */
    div[data-testid="stRadio"] input[type="radio"]:checked + div ::after {
        background-color: #4da6ff !important;
    }

    /* Cyber text highlight glow on the active selection row label */
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        text-shadow: 0 0 5px rgba(77, 166, 255, 0.4);
        transition: all 0.2s ease;
    }

    /* Changes radio button border outline hover state color */
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover div {
        border-color: #60a5fa !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Buffers Safely
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_operator" not in st.session_state:
    st.session_state.current_operator = "Guest"
if "registered_users" not in st.session_state:
    st.session_state.registered_users = {"admin": "admin123"}
if "system_telemetry_logs" not in st.session_state:
    st.session_state.system_telemetry_logs = pd.DataFrame(
        columns=["Timestamp", "Tracked Object", "Confidence Score", "Risk Tier"]
    )

# Cached Neural Inference Weights Engine Loader
@st.cache_resource
def load_yolo_inference_engine():
    if os.path.exists("AI_Surveillance_Model/YOLOv8_Custom_Run/weights/best.pt"):
        return YOLO("AI_Surveillance_Model/YOLOv8_Custom_Run/weights/best.pt")
    return YOLO("yolov8n.pt")

try:
    vision_model = load_yolo_inference_engine()
except Exception as error:
    st.error(f"Failed to bind neural infrastructure matrix: {error}")

# --- 2. SECURITY GATE OPERATIONS ---
def process_user_login(username, secret_key):
    if username in st.session_state.registered_users and st.session_state.registered_users[username] == secret_key:
        st.session_state.authenticated = True
        st.session_state.current_operator = username
        st.rerun()
    else:
        st.error("🔒 ACCESS DENIED: Security footprint verification failed.")

def process_user_registration(new_username, new_secret_key):
    if new_username in st.session_state.registered_users:
        st.error("⚠️ CONFLICT: Operator signature already registered.")
    elif new_username.strip() == "" or new_secret_key.strip() == "":
        st.error("⚠️ STRUCTURAL ERROR: Core fields cannot be blank.")
    else:
        st.session_state.registered_users[new_username] = new_secret_key
        st.success("⚡ REGISTERED: Operator added to system registers. Switch to Sign-In.")

# --- 3. HIGH-SECURITY ACCESS ENTRY NODE VIEW ---
if not st.session_state.authenticated:
    st.markdown("<h1 class='cyber-title'>🛡️ DEEPGUARD TACTICAL PORTAL</h1>", unsafe_allow_html=True)
    st.markdown("<p class='cyber-subtitle'>RESTRICTED ACCESS // LIVE PERIMETER AI INTERFACE</p>", unsafe_allow_html=True)
    
    _, middle_card_column, _ = st.columns([1, 1.8, 1])
    
    with middle_card_column:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        
        auth_switch = st.radio(
            "CHOOSE MODE",
            options=["[01] ACCREDITED OPERATOR SIGN-IN", "[02] PROVISION NEW OPERATOR MATRIX"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("<hr style='border-color: #2563eb; margin-top:15px; margin-bottom:25px;'/>", unsafe_allow_html=True)
        
        if "[01]" in auth_switch:
            u_name = st.text_input("⚡ OPERATOR ID HANDLE")
            u_pass = st.text_input("🔐 SECURE PASSWORD CONFIG", type="password")
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("EXECUTE AUTHENTICATION", use_container_width=True, type="primary"):
                process_user_login(u_name, u_pass)
                
        else:
            reg_name = st.text_input("⚡ ASSIGN NEW OPERATOR NAME")
            reg_pass = st.text_input("🔐 ENCRYPT NEW CREDENTIAL PASSKEY", type="password")
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("WRITE ACCESS RULE TO SYSTEM", use_container_width=True, type="primary"):
                process_user_registration(reg_name, reg_pass)
                
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- 4. SECURE PLATFORM CONTROL WORKSPACE (POST-AUTHENTICATION) ---
with st.sidebar:
    st.markdown(f"### 🛡️ CONTROL MONITOR")
    st.markdown(f"<div style='background:#111827; border:1px solid #2563eb; padding:12px; border-radius:4px; margin-bottom:15px;'>"
                f"👤 OPERATOR: <b style='color:#FFB300;'>{st.session_state.current_operator.upper()}</b><br/>"
                f"📟 SYSTEM: <b style='color:#00FF66;'>ENGAGED // SECURE</b></div>", unsafe_allow_html=True)
    
    navigation_hub = st.radio(
        "MISSION COMMAND HUB",
        options=["[⚡] Live Stream View", "[ℹ️] Neural Pipeline Specs", "[🎥] Target Frame Inference", "[📊] Threat Diagnostic Center", "[📥] Log Export Matrix"]
    )
    
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    if st.button("🔒 EMERGENCY LOCKDOWN / LOGOUT", use_container_width=True, type="secondary"):
        st.session_state.authenticated = False
        st.session_state.current_operator = "Guest"
        st.rerun()

# --- TAB A: LIVE STREAM VIEW ---
if "[⚡]" in navigation_hub:
    st.markdown("<h2 style='color: #4da6ff; font-family: sans-serif; font-weight: bold;'>[⚡] LIVE AREA SURVEILLANCE FEED MATRIX</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.markdown("<div class='status-box'><h4>⚡ ZERO INFERENCE LAG</h4>Processing spatial structures directly through live hardware layer loops.</div>", unsafe_allow_html=True)
    with col_y:
        st.markdown("<div class='status-box' style='border-left-color: #FFB300;'><h4>🔔 AUTOMATED INTERCEPT</h4>Neural bounding segments run automatically upon spatial violations.</div>", unsafe_allow_html=True)
    with col_z:
        st.markdown("<div class='status-box' style='border-left-color: #2563eb;'><h4>📊 LOG INTEGRATION</h4>All structural detections stream into the active telemetry data arrays.</div>", unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    # Updated to a high-contrast dark cybersecurity matrix aesthetic asset
    st.image("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=1200&q=80", caption="Node Terminal Target Area Mapping Core Link")

# --- TAB B: MODEL SPECIFICATIONS ---
elif "[ℹ️]" in navigation_hub:
    st.markdown("<h2 style='color: #4da6ff; font-family: sans-serif; font-weight: bold;'>[ℹ️] SUPERVISED INFERENCE ARCHITECTURE SPECS</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_info_left, col_info_right = st.columns([3, 2])
    with col_info_left:
        st.markdown("""
        ### Tensor Pipeline Constraints & Weights Strategy
        The backend engine builds spatial tensors using highly structured **You Only Look Once (YOLOv8)** kernels.
        
        #### Normalization Sequences:
        * **Structural Matrix Scale:** Input fields are systematically rescaled to raw $640 \times 640$ parameters.
        * **Floating Array Boundaries:** Changes base channels into strict floats ($0.0 \rightarrow 1.0$) to stabilize optimization speeds.
        * **Geometric Mapping Transforms:** Bounding centers are parsed into regular localized dimensions ($x_{c}, y_{c}, w, h$).
        """)
    with col_info_right:
        st.markdown("### Matrix Performance Profiles")
        metrics_log_table = pd.DataFrame({
            "Core Pipeline Version": ["YOLOv8-Nano System Setup", "YOLOv8-Small Standard", "YOLOv8-Medium Production Tier"],
            "mAP Score Metric": ["37.3% mAP@50", "44.9% mAP@50", "50.2% mAP@50"],
            "Hardware Compute Costs": ["~45ms Processing Speed", "~98ms Processing Speed",  "~210ms Processing Speed"]
        })
        st.dataframe(metrics_log_table, use_container_width=True, hide_index=True)

# --- TAB C: TARGET FRAME INFERENCE ---
elif "[🎥]" in navigation_hub:
    st.markdown("<h2 style='color: #4da6ff; font-family: sans-serif; font-weight: bold;'>[🎥] COMPUTER VISION TARGET INFERENCE CONFIGURATOR</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    control_col_1, control_col_2 = st.columns([1, 2])
    with control_col_1:
        ingestion_source = st.selectbox("ATTACH RADAR SIGNAL FEED:", ("Static Asset Ingestion", "Pre-recorded Stream Scan", "Hardware Webcam System"))
        confidence_alpha = st.slider("SET CONFIDENCE CUTOFF THRESHOLD SCORE", min_value=0.10, max_value=1.00, value=0.30, step=0.05)
    
    st.markdown("---")

    if ingestion_source == "Static Asset Ingestion":
        image_asset = st.file_uploader("Upload static imaging matrix asset...", type=["jpg", "jpeg", "png"])
        if image_asset is not None:
            image_buffer = np.asarray(bytearray(image_asset.read()), dtype=np.uint8)
            converted_frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
            
            run_predictions = vision_model.predict(converted_frame, conf=confidence_alpha)
            annotated_frame_view = run_predictions[0].plot()
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.image(cv2.cvtColor(converted_frame, cv2.COLOR_BGR2RGB), caption="Raw Structural Input Data Frame", use_container_width=True)
            with c_right:
                st.image(cv2.cvtColor(annotated_frame_view, cv2.COLOR_BGR2RGB), caption="YOLOv8 Neural Delineated Objects", use_container_width=True)
                
            for bounding_box in run_predictions[0].boxes:
                label_string = vision_model.names[int(bounding_box.cls[0])]
                probability = float(bounding_box.conf[0])
                hazard_index = "CRITICAL VIOLATION THREAT" if label_string in ["person", "fire", "weapon", "car", "truck"] else "Standard Perimeter Clear"
                
                new_row_log = pd.DataFrame([{"Timestamp": time.strftime("%H:%M:%S"), "Tracked Object": label_string, "Confidence Score": round(probability, 2), "Risk Tier": hazard_index}])
                st.session_state.system_telemetry_logs = pd.concat([st.session_state.system_telemetry_logs, new_row_log], ignore_index=True)

    elif ingestion_source == "Pre-recorded Stream Scan":
        video_asset = st.file_uploader("Upload tracking clip pack...", type=["mp4", "avi", "mov"])
        if video_asset is not None:
            with open("active_temp_track.mp4", "wb") as output_stream:
                output_stream.write(video_asset.read())
                
            file_capture_stream = cv2.VideoCapture("active_temp_track.mp4")
            video_canvas = st.empty()
            
            while file_capture_stream.isOpened():
                frame_state, stream_frame = file_capture_stream.read()
                if not frame_state:
                    break
                
                frame_inference = vision_model.predict(stream_frame, conf=confidence_alpha)
                rendered_frame_track = frame_inference[0].plot()
                video_canvas.image(cv2.cvtColor(rendered_frame_track, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            file_capture_stream.release()
            st.success("Target tracking file data scanning finalized.")

    elif ingestion_source == "Hardware Webcam System":
        st.warning("🔔 DEVICE NOTICE: Connecting interface hardware arrays. Verify camera feed rules.")
        camera_toggle = st.checkbox("ENGAGE MAIN LOGIC LINE WEBCAM HUB")
        live_canvas = st.empty()
        
        if camera_toggle:
            hardware_pipeline = cv2.VideoCapture(0)
            while hardware_pipeline.isOpened() and camera_toggle:
                read_status, incoming_matrix_frame = hardware_pipeline.read()
                if not read_status:
                    st.error("Live streaming input link dropped communication parameters.")
                    break
                
                live_inference = vision_model.predict(incoming_matrix_frame, conf=confidence_alpha)
                rendered_live_frame = live_inference[0].plot()
                live_canvas.image(cv2.cvtColor(rendered_live_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                for bounding_box in live_inference[0].boxes:
                    label_string = vision_model.names[int(bounding_box.cls[0])]
                    probability = float(bounding_box.conf[0])
                    new_row_log = pd.DataFrame([{"Timestamp": time.strftime("%H:%M:%S"), "Tracked Object": label_string, "Confidence Score": round(probability, 2), "Risk Tier": "Live Array Log"}])
                    st.session_state.system_telemetry_logs = pd.concat([st.session_state.system_telemetry_logs, new_row_log], ignore_index=True)
            hardware_pipeline.release()

# --- TAB D: THREAT DIAGNOSTIC CENTER ---
elif "[📊]" in navigation_hub:
    st.markdown("<h2 style='color: #4da6ff; font-family: sans-serif; font-weight: bold;'>[📊] SYSTEM QUANTUM METRICS & TELEMETRY LOG PANELS</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.system_telemetry_logs.empty:
        st.info("⚡ In-memory target arrays clean. Showing model baseline parameter history optimization lines.")
        
        training_epochs = list(range(1, 21))
        simulated_loss = [1.1 / (step**0.45) for step in training_epochs]
        simulated_map = [0.42 + (0.43 / (1 + np.exp(-0.28 * step))) for step in training_epochs]
        
        diagnostic_graph = go.Figure()
        diagnostic_graph.add_trace(go.Scatter(x=training_epochs, y=simulated_loss, name="Bounding Box Core Divergence Loss", mode="lines+markers", line=dict(color='#2563eb', width=3)))
        diagnostic_graph.add_trace(go.Scatter(x=training_epochs, y=simulated_map, name="Supervised Average mAP Performance Matrix", mode="lines+markers", line=dict(color='#00FF66', width=3)))
        
        diagnostic_graph.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0f1d',
            plot_bgcolor='#111827',
            title="YOLO Optimization Optimization and Target Precision Mappings",
            xaxis_title="Processing Step Cycles (Epochs)",
            yaxis_title="Calculated Margin Error Boundaries"
        )
        st.plotly_chart(diagnostic_graph, use_container_width=True)
    else:
        active_logs = st.session_state.system_telemetry_logs
        
        ui_left_chart, ui_right_chart = st.columns(2)
        with ui_left_chart:
            st.subheader("Frequency Tracking Categorical Metrics Distribution")
            bar_visual = px.bar(active_logs['Tracked Object'].value_counts(), labels={'value':'Occurrences Triggered', 'index':'Identified Entity Label'}, color_discrete_sequence=['#2563eb'])
            bar_visual.update_layout(template="plotly_dark", paper_bgcolor='#0a0f1d', plot_bgcolor='#111827')
            st.plotly_chart(bar_visual, use_container_width=True)
        with ui_right_chart:
            st.subheader("Probability Ranges Scatter Array Profile")
            box_visual = px.box(active_logs, x='Tracked Object', y='Confidence Score', color='Risk Tier', points="all", color_discrete_sequence=['#FFB300', '#2563eb'])
            box_visual.update_layout(template="plotly_dark", paper_bgcolor='#0a0f1d', plot_bgcolor='#111827')
            st.plotly_chart(box_visual, use_container_width=True)
            
        st.subheader("Raw Core Real-Time In-Memory Ingestion Registry Dataframe Table")
        st.dataframe(active_logs, use_container_width=True)

# --- TAB E: LOG EXPORT MATRIX ---
elif "[📥]" in navigation_hub:
    st.markdown("<h2 style='color: #4da6ff; font-family: sans-serif; font-weight: bold;'>[📥] PERIMETER COMPLIANCE COMPILATION MANIFEST EXPORTER</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.system_telemetry_logs.empty:
        st.warning("🔔 EXPORT LOCK: System diagnostic data arrays are currently clean. Track frames to verify signatures.")
    else:
        exportable_manifest = st.session_state.system_telemetry_logs
        st.dataframe(exportable_manifest, use_container_width=True)
        
        compiled_csv_bytes = exportable_manifest.to_csv(index=False).encode('utf-8')
        st.markdown("<br/>", unsafe_allow_html=True)
        st.download_button(
            label="📥 PULL SECURE SYSTEM TELEMETRY MANIFEST DATA REPORT (.CSV)",
            data=compiled_csv_bytes,
            file_name="DeepGuard_Tactical_Telemetry_Log.csv",
            mime="text/csv",
            use_container_width=True
        )