import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
from huggingface_hub import hf_hub_download

# Page setup
st.set_page_config(page_title="KrishiNetra · Plant Diagnostics", layout="wide", page_icon="https://github.com/Sanchal-01/KrishiNetra-AI/blob/main/01_app/Krishinetra_Logo.png?raw=true")

# This is how we have achieve Dark palette below using CSS additionally we have integrated Light mode is a new addition so switching modes doesn't touch a single existing dark-mode color or font-size.
PALETTES = {
    "dark": {
        "bg": "#10140F",
        "surface": "#171D15",
        "surface_2": "rgba(255,255,255,0.03)",
        "border": "rgba(237,234,224,0.08)",
        "border_strong": "rgba(237,234,224,0.18)",
        "text": "#F2F0E7",
        "text_muted": "#A6B2A3",
        "sage": "#63A67A",
        "sage_dim": "rgba(99,166,122,0.16)",
        "sage_line": "rgba(99,166,122,0.4)",
        "rust": "#D68A56",
        "rust_dim": "rgba(214,138,86,0.16)",
        "rust_line": "rgba(214,138,86,0.4)",
        "sidebar_bg": "#0D110B",
        "btn_text": "#0D140E",
        "btn_hover_bg": "#75B98D",
    },
    "light": {
        "bg": "#F7F5EF",
        "surface": "#FFFFFF",
        "surface_2": "rgba(30,42,31,0.035)",
        "border": "rgba(30,42,31,0.10)",
        "border_strong": "rgba(30,42,31,0.22)",
        "text": "#1E2A1F",
        "text_muted": "#5C6B5A",
        "sage": "#3F7350",
        "sage_dim": "rgba(63,115,80,0.10)",
        "sage_line": "rgba(63,115,80,0.4)",
        "rust": "#A85A2A",
        "rust_dim": "rgba(168,90,42,0.10)",
        "rust_line": "rgba(168,90,42,0.4)",
        "sidebar_bg": "#F1EEE4",
        "btn_text": "#F7F5EF",
        "btn_hover_bg": "#4E8A62",
    },
}

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

appearance = st.sidebar.radio(
    "APPEARANCE",
    ["Dark", "Light"],
    index=0 if st.session_state.theme_mode == "dark" else 1,
    horizontal=True,
)
st.session_state.theme_mode = appearance.lower()
p = PALETTES[st.session_state.theme_mode]

st.markdown(f"""
    <style>
    /* Fonts: Fraunces for the wordmark and result name, Inter for everything
       else, Plex Mono for the small lab-readout style labels/numbers */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {{
        --bg: {p['bg']};
        --surface: {p['surface']};
        --surface-2: {p['surface_2']};
        --border: {p['border']};
        --border-strong: {p['border_strong']};
        --text: {p['text']};
        --text-muted: {p['text_muted']};
        --sage: {p['sage']};
        --sage-dim: {p['sage_dim']};
        --sage-line: {p['sage_line']};
        --rust: {p['rust']};
        --rust-dim: {p['rust_dim']};
        --rust-line: {p['rust_line']};
        --sidebar-bg: {p['sidebar_bg']};
        --btn-text: {p['btn_text']};
        --btn-hover-bg: {p['btn_hover_bg']};
    }}

    html, body, [data-testid="stAppViewContainer"], .main {{
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
        /* crisper text on dark backgrounds, otherwise thin weights look fuzzy */
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }}

    div[data-testid="stAppViewBlockContainer"] {{
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        max-width: 1280px !important;
    }}

    /* the default Streamlit header and its colored top strip were showing
       through as a plain light bar, so pin both to the page background */
    [data-testid="stHeader"] {{
        background-color: var(--bg) !important;
    }}
    [data-testid="stDecoration"] {{
        background-image: none !important;
        background-color: var(--bg) !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border);
    }}

    /* Appearance toggle at the top of the sidebar */
    div[role="radiogroup"] {{
        gap: 6px;
    }}
    div[role="radiogroup"] label {{
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        padding: 4px 10px;
        background: var(--surface-2);
    }}
    div[role="radiogroup"] label p {{
        color: var(--text) !important;
    }}
    [data-testid="stWidgetLabel"] p {{
        color: var(--text) !important;
    }}

    /* Header */
    .masthead {{
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        border-bottom: 1px solid var(--border);
        padding-bottom: 18px;
        margin-bottom: 30px;
        flex-wrap: wrap;
        gap: 14px;
    }}
    .masthead-title {{
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        font-optical-sizing: auto;
        color: var(--text) !important;
        font-size: 44px !important;
        margin: 0 !important;
        letter-spacing: 0.2px;
        line-height: 1;
    }}
    .masthead-title em {{
        color: var(--sage);
        font-style: normal;
    }}
    .masthead-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.1px;
        margin-top: 6px;
    }}
    .masthead-status {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        font-weight: 500;
        letter-spacing: 0.6px;
        color: var(--text-muted);
        text-transform: uppercase;
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        padding: 7px 12px;
        background: var(--surface-2);
    }}
    .status-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--sage);
        box-shadow: 0 0 6px var(--sage);
    }}

    /* Section labels */
    .section-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--sage);
        margin-bottom: 4px;
    }}
    .section-heading {{
        font-family: 'Inter', sans-serif;
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 18.5px !important;
        margin-top: 0px !important;
        margin-bottom: 16px !important;
    }}

    /* Sidebar heading + note — previously inline hex, now themeable but
       same exact size/weight/spacing as before */
    .sidebar-heading {{
        color: var(--text) !important;
        font-family: 'Fraunces', serif;
        font-weight: 600;
        margin-top: 2px;
        font-size: 24px;
    }}
    .sidebar-note {{
        font-size: 14px !important;
        color: var(--text-muted) !important;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.2px;
        margin-top: 22px !important;
        line-height: 1.7;
        font-weight: 500;
    }}

    /* Sidebar ledger card */
    .ledger-card {{
        background-color: var(--surface-2);
        padding: 14px 15px;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-top: 16px;
    }}
    .ledger-row {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 13px;
        border-bottom: 1px dashed var(--border);
    }}
    .ledger-row:last-child {{ border-bottom: none; }}
    .ledger-key {{ color: var(--text-muted) !important; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.3px; font-weight: 500; }}
    .ledger-val {{ color: var(--text) !important; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }}

    /* Upload panel */
    .panel {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 22px 22px 22px;
    }}

    /* Streamlit's file uploader ships its own white surfaces for the
       dropzone and the uploaded-file row, so both need to be repainted
       to match the rest of the page */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] section {{
        background-color: var(--surface-2) !important;
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--sage-line) !important;
    }}
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"] {{
        background-color: transparent !important;
        color: var(--text) !important;
    }}
    [data-testid="stFileUploader"] small {{
        color: var(--text-muted) !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
    }}

    /* Select box (plant picker) */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background-color: var(--surface) !important;
        border-color: var(--border-strong) !important;
        color: var(--text) !important;
    }}

    /* Buttons */
    .stButton>button {{
        background: var(--sage) !important;
        color: var(--btn-text) !important;
        border-radius: 7px !important;
        padding: 11px 20px !important;
        font-weight: 700 !important;
        font-size: 14.5px !important;
        letter-spacing: 0.2px;
        border: none !important;
        width: 100% !important;
        transition: all 0.15s ease-in-out !important;
    }}
    .stButton>button:hover {{
        background: var(--btn-hover-bg) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px var(--sage-dim) !important;
    }}
    .stButton>button p {{
        color: var(--btn-text) !important;
    }}

    .stCheckbox label p {{ color: var(--text-muted) !important; font-size: 14px !important; }}

    /* Specimen result card — the one deliberately distinctive element,
       clipped like a lab specimen tag rather than a plain rounded card */
    .specimen-card {{
        position: relative;
        margin-top: 18px;
        padding: 20px 22px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--sage-line);
        border-radius: 10px;
    }}
    .specimen-card.is-warning {{
        border-left: 3px solid var(--rust-line);
    }}
    .specimen-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        color: var(--sage);
        display: block;
        margin-bottom: 10px;
    }}
    .specimen-card.is-warning .specimen-tag {{ color: var(--rust); }}
    .specimen-result {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 27px;
        color: var(--text);
        margin: 0 0 14px 0;
        line-height: 1.15;
    }}
    .specimen-meta {{
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }}
    .confidence-readout {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        font-weight: 600;
        color: var(--sage);
        background: var(--sage-dim);
        border: 1px solid var(--sage-line);
        padding: 6px 12px;
        border-radius: 5px;
        letter-spacing: 0.3px;
    }}
    .confidence-readout.is-warning {{
        color: var(--rust);
        background: var(--rust-dim);
        border-color: var(--rust-line);
    }}
    .plant-chip {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        font-weight: 500;
        color: var(--text-muted);
        letter-spacing: 0.4px;
    }}

    /* Idle / empty state */
    .empty-state {{
        border: 1px dashed var(--border-strong);
        border-radius: 10px;
        padding: 34px 20px;
        text-align: center;
        color: var(--text-muted);
        font-size: 14px;
        font-weight: 500;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.3px;
        margin-top: 6px;
    }}

    hr {{ border-color: var(--border) !important; }}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<div class='section-eyebrow'>Control Panel</div>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 class='sidebar-heading'>Field Ledger</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

plant_choice = st.sidebar.selectbox(
    "Select the plant type:",
    ["Apple", "Grapes", "Potato", "Corn", "Tomato"]
)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

HF_REPO_ID = "Sanchal-01/krishi_netra-models"


MODEL_MAP = {
    "Apple": {
        "path": "CNN_MODELS/model_apple_21.1.8.keras",
        "labels": ["Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Healthy"]
    },
    "Grapes": {
        "path": "CNN_MODELS/model_grapes_1.0.2.keras",
        "labels": ["Grape Black Rot", "Grape_Esca (Black Measles)", "Grape_Leaf Blight", "Healthy"]
    },
    "Potato": {
        "path": "CNN_MODELS/potato_model_18.keras",
        "labels": ["Potato_Early Blight", "Potato_Late Blight", "Healthy"]
    },
    "Corn": {
        "path": "CNN_MODELS/corn_model_11.keras",
        "labels": ['Corn Blight', 'Corn_Common Rust', 'Corn_Gray Leaf Spot', 'Healthy']
    },
    "Tomato": {
        "path": "CNN_MODELS/model_tomato_1.0.1.keras",
        "labels": ["Tomato_Late Blight", "Tomato_Leaf Mold", "Tomato_Septoria Leaf Spot", "Healthy"]
    }
}

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# @st.cache_resource
# def load_expert_model(path):
#     return tf.keras.models.load_model(path)

@st.cache_resource
def load_expert_model(filename):
    # pulls the file from the private HF repo on first use, then Streamlit's  own cache keeps it in memory so this only runs once per model per session
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=st.secrets["HF_TOKEN"],
    )
    return tf.keras.models.load_model(local_path)
 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

st.sidebar.markdown(f"""
    <div class='ledger-card'>
        <div class='ledger-row'><span class='ledger-key'>ENGINE</span><span class='ledger-val'>ACTIVE</span></div>
        <div class='ledger-row'><span class='ledger-key'>MODEL</span><span class='ledger-val'>Dedicated CNN</span></div>
        <div class='ledger-row'><span class='ledger-key'>SPECIMEN</span><span class='ledger-val'>{plant_choice}</span></div>
        <div class='ledger-row'><span class='ledger-key'>SCOPE</span><span class='ledger-val'>{len(MODEL_MAP[plant_choice]["labels"])} classes</span></div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <p class='sidebar-note'>
    Each plant is served by a purpose-trained model rather than one shared classifier — narrower scope, sharper reads.
    </p>
""", unsafe_allow_html=True)




# Masthead
st.markdown("""
    <div class='masthead'>
        <div>
            <h1 class='masthead-title'>Krishi<em>Netra</em></h1>
            <div class='masthead-tag'>Multi-Plant Disease Diagnostic Engine</div>
        </div>
        <div class='masthead-status'>
            <span class='status-dot'></span> Model Ready
        </div>
    </div>
""", unsafe_allow_html=True)

# Workspace
col1, col2 = st.columns([1.1, 0.9], gap="large")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

with col1:
    st.markdown("<div class='section-eyebrow'>Step 01</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Image Input</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload leaf image...",
        type=["jpg", "jpeg", "png", "jfif"],
        label_visibility="collapsed"
    )

    image_to_predict = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        use_crop = st.checkbox("Enable cropping workspace", value=False)
        st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)

        if use_crop:
            image_to_predict = st_cropper(
                image,
                realtime_update=True,
                box_color="#B72018",
                aspect_ratio=None,
                return_type="image"
            )
            st.image(image_to_predict, caption="Target crop segment", width=400)
        else:
            st.image(image, caption="Full source leaf image", width=400)
            image_to_predict = image

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

with col2:
    st.markdown("<div class='section-eyebrow'>Step 02</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>Diagnostics Report</div>", unsafe_allow_html=True)

    if uploaded_file is not None and image_to_predict is not None:
        if st.button("Run Diagnosis"):
            with st.spinner("Analyzing tissue patterns, please wait..."):

                model_path = MODEL_MAP[plant_choice]["path"]
                current_labels = MODEL_MAP[plant_choice]["labels"]

                try:
                    model = load_expert_model(model_path)
                    image_rgb = image_to_predict.convert("RGB")
                    img = image_rgb.resize((224, 224))

                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = model.predict(img_array)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    result = current_labels[predicted_class_idx]

                    if confidence >= 70.0:
                        st.markdown(f"""
                            <div class='specimen-card'>
                                <span class='specimen-tag'>Diagnosis · High Confidence</span>
                                <h3 class='specimen-result'>{result}</h3>
                                <div class='specimen-meta'>
                                    <span class='confidence-readout'>CONFIDENCE {confidence:.2f}%</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='specimen-card is-warning'>
                                <span class='specimen-tag'>Diagnosis · Low Confidence</span>
                                <h3 class='specimen-result'>{result}</h3>
                                <div class='specimen-meta'>
                                    <span class='confidence-readout is-warning'>CONFIDENCE {confidence:.2f}%</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.info(
                            "Low confidence may indicate mixed symptoms. For a sharper read, crop tightly at the disease or to a single leaf."
                        )

                except OSError:
                    st.error(f"Model file not found at: `{model_path}`")
                except Exception as e:
                    st.error(f"Error during processing: {e}")
    else:
        st.markdown("""
            <div class='empty-state'>
                AWAITING SPECIMEN — UPLOAD AN IMAGE TO BEGIN ANALYSIS
            </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #        

# Footer disclaimer
st.markdown("""
    <div style='margin-top:40px; padding-top:16px; border-top:1px solid var(--border);
                text-align:center; font-family:IBM Plex Mono, monospace; font-size:12px;
                font-weight:500; letter-spacing:0.3px; color:var(--text-muted);'>
        KrishiNetra is an AI model and can make mistakes - results may be inaccurate for blurry,
        poorly lit, or unfamiliar images. Please verify with an agricultural expert before acting on a diagnosis.
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #