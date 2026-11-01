from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_UPLOAD_MB = 10
IMAGE_SIZE = 224  # Match the image size used for the trained classifier.

st.set_page_config(
    page_title="LeafGuard AI | Plant Health Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --lg-bg: #07130f;
        --lg-panel: #10221a;
        --lg-panel-soft: #142b20;
        --lg-border: rgba(168, 220, 187, 0.14);
        --lg-text: #edf7f0;
        --lg-muted: #9ab4a3;
        --lg-green: #69d899;
        --lg-green-dark: #1e6b48;
      }

      .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(40, 120, 75, .20), transparent 34%),
          radial-gradient(ellipse at 95% 8%, rgba(46, 111, 77, .13), transparent 28%),
          var(--lg-bg);
        color: var(--lg-text);
      }

      .block-container {
        max-width: 1120px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
      }

      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { right: 1rem; }

      .lg-brand-row {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin-bottom: 1.7rem;
      }
      .lg-brand-mark {
        width: 44px;
        height: 44px;
        border: 1px solid rgba(105, 216, 153, .34);
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(145deg, rgba(105,216,153,.19), rgba(105,216,153,.05));
        font-size: 1.45rem;
      }
      .lg-brand-name {
        color: var(--lg-text);
        font-size: 1.06rem;
        font-weight: 750;
        letter-spacing: .01em;
        line-height: 1.15;
      }
      .lg-brand-sub {
        color: var(--lg-muted);
        font-size: .76rem;
        margin-top: .18rem;
      }
      .lg-status {
        margin-left: auto;
        border: 1px solid rgba(105,216,153,.25);
        color: #9ce7b9;
        background: rgba(39, 119, 75, .16);
        border-radius: 999px;
        padding: .35rem .7rem;
        font-size: .72rem;
        font-weight: 650;
        letter-spacing: .04em;
      }

      .lg-hero {
        border: 1px solid var(--lg-border);
        border-radius: 24px;
        padding: 2rem 2rem 1.85rem;
        margin-bottom: 1.35rem;
        background: linear-gradient(120deg, rgba(21, 54, 37, .94), rgba(12, 31, 23, .92));
        box-shadow: 0 18px 55px rgba(0,0,0,.16);
      }
      .lg-eyebrow {
        color: #9de3b8;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .69rem;
        font-weight: 750;
        margin-bottom: .8rem;
      }
      .lg-hero h1 {
        color: #f1faf4;
        font-size: clamp(2rem, 4.4vw, 3.25rem);
        letter-spacing: -.045em;
        line-height: 1.05;
        margin: 0 0 .8rem 0;
        font-weight: 780;
      }
      .lg-hero p {
        color: #bfd7c7;
        max-width: 650px;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
      }

      .lg-section-title {
        color: var(--lg-text);
        font-size: 1.08rem;
        font-weight: 720;
        margin: .25rem 0 .35rem;
      }
      .lg-section-copy {
        color: var(--lg-muted);
        font-size: .86rem;
        line-height: 1.5;
        margin-bottom: 1rem;
      }
      .lg-panel {
        border: 1px solid var(--lg-border);
        border-radius: 19px;
        background: rgba(15, 34, 25, .78);
        padding: 1.15rem;
      }
      .lg-result-label {
        color: var(--lg-muted);
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: .68rem;
        font-weight: 750;
      }
      .lg-result-value {
        font-size: 1.8rem;
        line-height: 1.2;
        font-weight: 800;
        margin: .42rem 0 .22rem;
        letter-spacing: -.035em;
      }
      .lg-result-healthy { color: #80e6a8; }
      .lg-result-unhealthy { color: #ffb27d; }
      .lg-small-muted {
        color: var(--lg-muted);
        font-size: .78rem;
        line-height: 1.55;
      }
      .lg-footer {
        border-top: 1px solid var(--lg-border);
        margin-top: 2.3rem;
        padding-top: 1rem;
        color: #829b8b;
        font-size: .74rem;
        display: flex;
        justify-content: space-between;
        gap: .75rem;
        flex-wrap: wrap;
      }

      div[data-testid="stFileUploader"] section {
        border: 1px dashed rgba(105,216,153,.38);
        border-radius: 16px;
        background: rgba(105,216,153,.035);
      }
      div[data-testid="stFileUploader"] section:hover {
        border-color: rgba(105,216,153,.75);
      }
      div.stButton > button[kind="primary"] {
        border: 1px solid #65d394;
        background: linear-gradient(110deg, #28784f, #1d6040);
        color: #f5fff8;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 750;
        box-shadow: 0 8px 22px rgba(20, 103, 61, .18);
      }
      div.stButton > button[kind="primary"]:hover {
        border-color: #9ce7b9;
        background: linear-gradient(110deg, #328b5d, #24744c);
        color: white;
      }
      div[data-testid="stProgressBar"] > div > div {
        background-color: #69d899;
      }
      .stAlert { border-radius: 12px; }
      @media (max-width: 700px) {
        .block-container { padding-top: 1.2rem; }
        .lg-hero { padding: 1.45rem; border-radius: 18px; }
        .lg-status { display: none; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model loading and prediction helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the LeafGuard AI model...")
def load_model(model_path: str) -> YOLO:
    """Load the trained YOLO classification checkpoint once per app process."""
    return YOLO(model_path)


def prepare_image(file_bytes: bytes) -> Image.Image:
    """Decode, orient and convert the uploaded image to RGB."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file could not be read as a valid image.") from exc


def run_classification(model: YOLO, image: Image.Image) -> dict[str, Any]:
    """Run CPU inference and return the two class probabilities."""
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    if not results or results[0].probs is None:
        raise RuntimeError("The model did not return classification probabilities.")

    result = results[0]
    probabilities = result.probs.data.detach().cpu().numpy().astype(float)
    names = result.names

    # Resolve class indices by label rather than assuming a fixed class order.
    class_indices = {
        str(name).strip().casefold(): int(index)
        for index, name in names.items()
    }

    if "healthy" not in class_indices or "unhealthy" not in class_indices:
        raise RuntimeError(
            "The checkpoint must contain exactly the expected 'Healthy' and "
            "'Unhealthy' class labels. Check that you selected the correct best.pt."
        )

    healthy_probability = float(probabilities[class_indices["healthy"]])
    unhealthy_probability = float(probabilities[class_indices["unhealthy"]])
    predicted_index = int(np.argmax(probabilities))
    predicted_label = str(names[predicted_index]).strip()

    return {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_index]),
        "healthy_probability": healthy_probability,
        "unhealthy_probability": unhealthy_probability,
    }


# -----------------------------------------------------------------------------
# Page content
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="lg-brand-row">
      <div class="lg-brand-mark">🌿</div>
      <div>
        <div class="lg-brand-name">LeafGuard AI</div>
        <div class="lg-brand-sub">Plant health classification</div>
      </div>
      <div class="lg-status">YOLO26 · IMAGE CLASSIFICATION</div>
    </div>

    <section class="lg-hero">
      <div class="lg-eyebrow">AI-powered plant screening</div>
      <h1>A clearer view of<br>plant health.</h1>
      <p>
        Upload a photograph of a plant leaf to classify it as healthy or
        unhealthy using the LeafGuard AI image-classification model.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("The trained model file was not found.")
    st.markdown(
        "Place your trained **`best.pt`** file inside the **`models`** folder "
        "so the path is `models/best.pt`, then restart the app."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:  # noqa: BLE001 - show an actionable UI error.
    st.error("LeafGuard AI could not load the model checkpoint.")
    st.code(str(exc))
    st.stop()

left, right = st.columns([1.02, 0.98], gap="large")

with left:
    st.markdown('<div class="lg-section-title">Upload a leaf image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lg-section-copy">Choose a clear, well-lit photograph. '
        'Supported formats: JPG, JPEG and PNG.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=False,
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
        label_visibility="collapsed",
    )

    image: Image.Image | None = None
    image_digest: str | None = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
        else:
            try:
                image = prepare_image(file_bytes)
                image_digest = hashlib.sha256(file_bytes).hexdigest()
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )
                st.caption(f"Image dimensions: {image.width:,} × {image.height:,} pixels")
            except ValueError as exc:
                st.error(str(exc))

    if image_digest and st.session_state.get("image_digest") != image_digest:
        st.session_state.pop("analysis", None)
        st.session_state["image_digest"] = image_digest

    analyze_clicked = st.button(
        "Analyse leaf",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )

    if analyze_clicked and image is not None:
        try:
            with st.spinner("Analysing leaf image..."):
                st.session_state["analysis"] = run_classification(model, image)
        except Exception as exc:  # noqa: BLE001 - keep inference errors user-readable.
            st.session_state.pop("analysis", None)
            st.error("The image could not be analysed. Please try another image.")
            st.caption(f"Technical detail: {exc}")

with right:
    st.markdown('<div class="lg-section-title">Classification result</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lg-section-copy">The model returns one of two classes '
        'and a probability for each class.</div>',
        unsafe_allow_html=True,
    )

    analysis = st.session_state.get("analysis")

    if analysis:
        label = analysis["label"]
        is_healthy = label.casefold() == "healthy"
        result_class = "lg-result-healthy" if is_healthy else "lg-result-unhealthy"
        symbol = "✓" if is_healthy else "!"
        confidence_percent = analysis["confidence"] * 100

        st.markdown(
            f"""
            <div class="lg-panel">
              <div class="lg-result-label">Model prediction</div>
              <div class="lg-result-value {result_class}">{symbol} {label}</div>
              <div class="lg-small-muted">Top-class probability: {confidence_percent:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("**Class probabilities**")

        healthy_percent = analysis["healthy_probability"] * 100
        unhealthy_percent = analysis["unhealthy_probability"] * 100

        st.markdown(
            f"<div class='lg-small-muted'>Healthy <b style='float:right;color:#80e6a8'>{healthy_percent:.2f}%</b></div>",
            unsafe_allow_html=True,
        )
        st.progress(analysis["healthy_probability"])

        st.markdown(
            f"<div class='lg-small-muted'>Unhealthy <b style='float:right;color:#ffb27d'>{unhealthy_percent:.2f}%</b></div>",
            unsafe_allow_html=True,
        )
        st.progress(analysis["unhealthy_probability"])

        st.info(
            "This is an image-based screening result, not a definitive plant "
            "disease diagnosis. Prediction probabilities are not calibrated "
            "measures of real-world certainty."
        )
    else:
        st.markdown(
            """
            <div class="lg-panel" style="min-height:250px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;">
              <div style="font-size:2.4rem;margin-bottom:.8rem;">🍃</div>
              <div style="font-size:1.05rem;font-weight:750;color:#edf7f0;margin-bottom:.35rem;">Ready when you are</div>
              <div class="lg-small-muted" style="max-width:280px;">Upload a leaf photograph and select Analyse leaf to see the classification.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="lg-footer">
      <span>LeafGuard AI · YOLO26 binary image classifier</span>
      <span>For educational and screening use</span>
    </div>
    """,
    unsafe_allow_html=True,
)
