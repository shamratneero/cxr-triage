"""
Streamlit UI for the AI-Assisted Chest X-Ray Triage System.

THIS is the actual interface doctors/radiologists will use during the pilot
study — NOT the FastAPI /docs page (that was only for us to test the backend).

RUN WITH:
  cd D:\\cxr-triage
  streamlit run streamlit_app.py

REQUIRES: app.py (the FastAPI backend) must already be running separately in
another terminal window:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Chest X-Ray Triage Assistant",
    page_icon="🫁",
    layout="wide",
)

# ─── Tier colors/labels — used for the big priority badge ──────────────
TIER_STYLE = {
    "critical": {"color": "#B00020", "bg": "#FDE7E9", "label": "CRITICAL — Review Immediately"},
    "urgent":   {"color": "#B45F06", "bg": "#FFF3E0", "label": "URGENT — Review Soon"},
    "routine":  {"color": "#1B5E20", "bg": "#E8F5E9", "label": "ROUTINE — Standard Queue"},
}


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_predict(image_bytes, filename, content_type):
    files = {"file": (filename, image_bytes, content_type)}
    response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def render_tier_badge(tier):
    style = TIER_STYLE.get(tier, TIER_STYLE["routine"])
    st.markdown(
        f"""
        <div style="
            background-color: {style['bg']};
            color: {style['color']};
            padding: 20px;
            border-radius: 10px;
            border: 2px solid {style['color']};
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 20px;
        ">
            {style['label']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_findings_table(per_finding_detail):
    flagged = [f for f in per_finding_detail if f['flagged']]
    not_flagged = [f for f in per_finding_detail if not f['flagged']]

    if flagged:
        st.subheader("Flagged Findings")
        for f in flagged:
            pct = f['calibrated_confidence'] * 100
            urgency = f['clinical_urgency']
            urgency_color = TIER_STYLE.get(urgency, TIER_STYLE["routine"])['color']
            st.markdown(
                f"**{f['label']}** — {pct:.0f}% confidence "
                f"&nbsp;<span style='color:{urgency_color}; font-weight:bold;'>[{urgency.upper()}]</span>",
                unsafe_allow_html=True,
            )
            st.progress(min(f['calibrated_confidence'], 1.0))
    else:
        st.info("No findings crossed their confidence threshold for this image.")

    with st.expander("See all 14 findings (including negative)"):
        for f in not_flagged:
            pct = f['calibrated_confidence'] * 100
            st.caption(f"{f['label']}: {pct:.0f}% (threshold {f['threshold']*100:.0f}%) — not flagged")


def render_heatmaps(heatmaps_base64):
    if not heatmaps_base64:
        return
    st.subheader("Where the model is looking")
    st.caption(
        "Red/yellow regions show where the AI focused when detecting each flagged finding. "
        "Use this to sanity-check whether the highlighted area matches the actual finding location."
    )
    cols = st.columns(len(heatmaps_base64))
    for col, (label, b64_str) in zip(cols, heatmaps_base64.items()):
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes))
        with col:
            st.image(img, caption=label, use_container_width=True)


# ─── Main page ────────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Triage Assistant")
st.caption(
    "Pilot research tool — AI-assisted triage prioritization. "
    "Not a diagnostic device. All results must be reviewed by a qualified clinician."
)

if not check_api_health():
    st.error(
        "⚠️ Cannot reach the analysis server. Make sure the backend is running:\n\n"
        "`uvicorn app:app --host 0.0.0.0 --port 8000 --reload`\n\n"
        "in a separate terminal window, then refresh this page."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Accepts PNG or JPEG chest X-ray images."
)

if uploaded_file is not None:
    col_img, col_results = st.columns([1, 1.3])

    with col_img:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_container_width=True)

    with st.spinner("Analyzing image..."):
        try:
            image_bytes = uploaded_file.getvalue()
            result = call_predict(image_bytes, uploaded_file.name, uploaded_file.type)
        except requests.exceptions.RequestException as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    with col_results:
        render_tier_badge(result['case_tier'])

        if result['driving_findings']:
            st.markdown(
                f"**Priority driven by:** {', '.join(result['driving_findings'])}"
            )

        render_findings_table(result['per_finding_detail'])

    st.divider()
    render_heatmaps(result.get('heatmaps_base64', {}))

    with st.expander("Model details"):
        st.json(result['model_info'])

else:
    st.info("👆 Upload a chest X-ray image to begin analysis.")

st.divider()
st.caption(
    "⚠️ Research prototype under pilot clinical validation. "
    "This tool is intended to assist triage prioritization only and does not "
    "constitute a medical diagnosis. All flagged and unflagged findings should "
    "be independently reviewed by a qualified radiologist or physician."
)
