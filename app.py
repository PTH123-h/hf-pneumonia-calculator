import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HF vs Pneumonia Calculator", layout="centered")


@st.cache_resource
def load_model():
    obj = joblib.load("model.joblib")  # {"model":..., "features":[...]}
    return obj["model"], obj["features"]


def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as r:
            return json.load(r)
    except Exception:
        return default


model, FEATURES = load_model()
config = load_json("config.json", default={})

POSITIVE_CLASS_NAME = "Heart Failure"   # y=1
NEGATIVE_CLASS_NAME = "Pneumonia"       # y=0
CUTOFF = float(config.get("youden_threshold", 0.50))

st.markdown("""
<style>
/* Layout */
.block-container { max-width: 920px; padding-top: 1.2rem; padding-bottom: 0.8rem; }
h1 { margin: 0 0 0.2rem 0; font-size: 1.6rem; line-height: 1.25; }

/* Custom title: avoids top-cropping and supports wrapping */
.app-title {
  font-size: 1.65rem;
  font-weight: 800;
  line-height: 1.25;
  margin: 0.1rem 0 0.35rem 0;
  white-space: normal;
  word-break: break-word;
}

.subtitle { color:#555; font-size:0.95rem; line-height:1.35; margin: 0 0 0.6rem 0; }
.section-title { font-size: 1.05rem; font-weight: 750; margin: 0.4rem 0 0.35rem 0; }
.result-card { border:1px solid #e6e6e6; border-radius:12px; padding:12px 14px; background:#fff; margin-top: 0.6rem; }
.small { color:#666; font-size:0.88rem; line-height:1.35; margin-top: 0.25rem; }
hr { margin: 0.6rem 0; }
</style>
""", unsafe_allow_html=True)


def predict_proba_hf(x_df: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_df)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return float(proba[0, classes.index(1)])
        return float(proba[0, 1])
    score = float(model.decision_function(x_df)[0])
    return float(1.0 / (1.0 + np.exp(-score)))


# --- Title + subtitle (use custom HTML title to prevent cropping) ---
st.markdown(
    '<div class="app-title">Online Model-Based Calculator to Differentiate Heart Failure from Pneumonia</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Model-based estimates for <b>Heart Failure</b> (y=1) versus <b>Pneumonia</b> '
    'using Age, AG, CREA, UA, RDW, and PDW.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="section-title">Inputs</div>', unsafe_allow_html=True)

with st.form("hf_pna_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age (years)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0, format="%.0f")
        ag = st.number_input("AG (mmol/L)", min_value=0.0, max_value=2000.0, value=16.0, step=0.1, format="%.1f")
        crea = st.number_input("CREA (µmol/L)", min_value=0.0, max_value=500000.0, value=90.0, step=1.0, format="%.0f")

    with c2:
        ua = st.number_input("UA (µmol/L)", min_value=0.0, max_value=500000.0, value=350.0, step=1.0, format="%.0f")
        rdw = st.number_input("RDW (%)", min_value=0.0, max_value=2000.0, value=13.5, step=0.1, format="%.1f")
        pdw = st.number_input("PDW (fL)", min_value=0.0, max_value=2000.0, value=12.5, step=0.1, format="%.1f")

    submitted = st.form_submit_button("Predict")

if submitted:
    x = pd.DataFrame([{
        "Age": age, "AG": ag, "CREA": crea, "UA": ua, "RDW": rdw, "PDW": pdw
    }])[FEATURES]

    p_hf = predict_proba_hf(x)
    p_pna = 1.0 - p_hf
    pred_label = POSITIVE_CLASS_NAME if p_hf >= CUTOFF else NEGATIVE_CLASS_NAME

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("**Estimated probabilities (binary)**")
    st.markdown(f"- {POSITIVE_CLASS_NAME}: **{p_hf:.2f}**")
    st.markdown(f"- {NEGATIVE_CLASS_NAME}: **{p_pna:.2f}**")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"**Model decision (classification):** **{pred_label}**")
    st.markdown(
        f"<div class='small'>"
        f"Classification is based on P(HF) with a fixed threshold (t = {CUTOFF:.3f})."
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
