import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "best_densenet.keras")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_image")
SAMPLE_PATHS = [
    os.path.join(SAMPLE_DIR, filename)
    for filename in sorted(os.listdir(SAMPLE_DIR))
    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
] if os.path.isdir(SAMPLE_DIR) else []
SAMPLE_IMAGE_PATH = SAMPLE_PATHS[0] if SAMPLE_PATHS else None

print("Current directory:", os.getcwd())
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

st.set_page_config(
    page_title="PneumoniaCare AI",
    page_icon="🩻",
    layout="wide",
)

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

model = load_model()


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No convolutional layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def apply_gradcam_overlay(original_image, heatmap, alpha=0.55):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(original_image.size)
    heatmap_np = np.array(heatmap)

    colored = np.zeros((heatmap_np.shape[0], heatmap_np.shape[1], 3), dtype=np.uint8)
    colored[..., 0] = np.clip(heatmap_np * 1.0, 0, 255)
    colored[..., 1] = np.clip(heatmap_np * 0.8, 0, 255).astype(np.uint8)
    colored[..., 2] = np.clip(255 - heatmap_np, 0, 255)

    colored = Image.fromarray(colored)
    return Image.blend(original_image.convert("RGB"), colored, alpha=alpha)

page_style = """
<style>
body {
    background: #f3f7fb;
    color: #0f172a;
}
section.main {
    padding-top: 0rem;
}
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    padding: 22px 28px;
    border-radius: 28px;
    background: linear-gradient(135deg, #0f4c81 0%, #1a73b8 100%);
    color: white;
    box-shadow: 0 25px 70px rgba(15, 23, 42, 0.18);
    margin-bottom: 24px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 16px;
}
.brand-icon {
    width: 56px;
    height: 56px;
    border-radius: 18px;
    background: rgba(255,255,255,0.18);
    display: grid;
    place-items: center;
    font-size: 1.35rem;
    font-weight: 800;
}
.brand-text {
    line-height: 1.05;
}
.brand-text .title {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
}
.brand-text .subtitle {
    color: rgba(255,255,255,0.88);
    font-size: 0.98rem;
}
.nav-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 12px 24px;
    margin-right: 12px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.24);
    background: rgba(255,255,255,0.12);
    color: white;
    font-weight: 600;
    cursor: pointer;
}
.nav-pill.active {
    background: white;
    color: #0f4c81;
}
.top-actions {
    display: flex;
    align-items: center;
    gap: 12px;
}
.action-dot {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: rgba(255,255,255,0.18);
    display: grid;
    place-items: center;
    color: white;
    font-size: 1.05rem;
    cursor: pointer;
}
.content-card,
.side-card,
.info-card,
.small-card,
.alert-card {
    border-radius: 26px;
    padding: 28px;
    box-shadow: 0 20px 48px rgba(15, 23, 42, 0.08);
    margin-bottom: 22px;
}
.content-card,
.side-card,
.info-card,
.small-card {
    background: white;
}
.alert-card {
    background: linear-gradient(135deg, #fee2e2 0%, #f8d7da 100%);
    border: 1px solid #f5c2c7;
    color: #842029;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 18px;
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
}
.stat-card {
    border-radius: 22px;
    background: linear-gradient(135deg, #0f4c81 0%, #1a73b8 100%);
    color: white;
    padding: 20px;
}
.stat-card .label {
    color: rgba(255,255,255,0.84);
    margin-bottom: 10px;
}
.stat-card .value {
    font-size: 2rem;
    font-weight: 700;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 16px;
}
.status-pill.good { background: #d1fae5; color: #166534; }
.status-pill.warning { background: #fef3c7; color: #92400e; }
.status-pill.bad { background: #fee2e2; color: #b91c1c; }
.small-card {
    padding: 18px;
    background: #f8fbff;
}
.small-card strong {
    display: block;
    margin-bottom: 8px;
    color: #0f172a;
}
.small-card p {
    margin: 0;
    color: #475569;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

st.markdown(
    "<div class='topbar'>"
    "<div class='brand'>"
    "<div class='brand-icon'>M+</div>"
    "<div class='brand-text'>"
    "<div class='title'>MediScan</div>"
    "<div class='subtitle'>Pneumonia classification dashboard</div>"
    "</div>"
    "</div>"
    "</div>"
    , unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([3, 1])
with hero_left:
    st.markdown(
        "<div class='content-card'>"
        "<div class='section-title'>Overview conditions</div>"
        "<p style='line-height:1.8; color:#334155;'>"
        "Evaluate chest X-ray scans using a secure healthcare interface. Review prediction confidence, scan details, and Grad-CAM explanations in one polished dashboard."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        "<div class='content-card'>"
        "<div class='section-title'>Clinical summary</div>"
        "<p style='line-height:1.8; color:#334155;'>"
        "A high-quality, clinical-style interface for pneumonia screening and X-ray review."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader(
    "Upload Chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray for model inference.",
)
sample_button = st.button("Use sample X-ray")

sample_image = None
sample_file_used = False
if sample_button:
    sample_file_used = True
    sample_image_path = next((p for p in SAMPLE_PATHS if os.path.exists(p)), None)
    if sample_image_path:
        sample_image = load_image(sample_image_path).convert("RGB")
    else:
        st.warning("No sample image found in the App folder.")

if uploaded_file is None and sample_image is None:
    st.info("Upload an image or use the example X-ray button to begin.")
else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = sample_image

    main_panel = st.container()

    with main_panel:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Scan workspace</div>", unsafe_allow_html=True)
        st.image(image, caption="Chest X-ray preview", width=640)

        st.markdown("<div class='section-title' style='margin-top:24px;'>Results</div>", unsafe_allow_html=True)
        with st.spinner("Analyzing image..."):
            img = image.resize((224, 224))
            img = np.array(img).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img, verbose=0)[0][0]
            probability = float(prediction)

        if probability > 0.5:
            status_text = "PNEUMONIA"
            status_class = "bad"
        else:
            status_text = "NORMAL"
            status_class = "good"

        result_col, advice_col = st.columns([3, 1])
        with result_col:
            st.markdown(
                f"<div class='status-pill {status_class}'>Predicted: {status_text}</div>",
                unsafe_allow_html=True,
            )
            st.metric("Pneumonia probability", f"{probability * 100:.2f}%")

            if st.checkbox("Show Grad-CAM overlay", value=True):
                heatmap = make_gradcam_heatmap(img, model)
                overlay = apply_gradcam_overlay(image, heatmap)
                st.markdown("<div class='section-title' style='margin-top:24px;'>Grad-CAM explanation</div>", unsafe_allow_html=True)
                grad_view1, grad_view2 = st.columns(2)
                grad_view1.image(image, caption="Original X-ray", width=300)
                grad_view2.image(overlay, caption="Grad-CAM overlay", width=300)

        with advice_col:
            if probability > 0.5:
                st.markdown(
                    "<div class='alert-card' style='background: linear-gradient(135deg, #fda4af 0%, #fecdd3 100%); border-color: #f87171; color: #7f1d1d;'>"
                    "<div class='section-title'>Pneumonia care guidance</div>"
                    "<p style='margin:0 0 12px; font-weight:600;'>Please consult a doctor immediately.</p>"
                    "<p style='margin:0 0 16px;'>These tips are supportive care suggestions and should be combined with professional medical advice.</p>"
                    "<ul style='margin: 0; padding-left: 18px; color: #7f1d1d;'>"
                    "<li>Rest and stay well hydrated to support recovery.</li>"
                    "<li>Use a humidifier or warm fluids to ease breathing and chest comfort.</li>"
                    "<li>Follow prescribed medications and complete the full treatment course.</li>"
                    "</ul>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='alert-card' style='background: linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%); border-color: #cbd5e1; color: #0f172a;'>"
                    "<div class='section-title'>Guidance</div>"
                    "<p style='margin:0;'>The model did not detect pneumonia. Continue monitoring symptoms and consult a clinician if needed.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        details_col, samples_col = st.columns([2, 1])
        with details_col:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Scan details</div>", unsafe_allow_html=True)
            st.write(f"**Raw score:** {probability:.4f}")
            st.write("**Decision threshold:** 0.50")
            st.write(f"**Source:** {'Sample image' if sample_file_used else 'Uploaded file'}")
            st.markdown("</div>", unsafe_allow_html=True)

        with samples_col:
            if SAMPLE_PATHS:
                st.markdown("<div class='side-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Sample X-rays</div>", unsafe_allow_html=True)
                for sample_path in SAMPLE_PATHS[:3]:
                    preview_image = load_image(sample_path).convert("RGB")
                    st.image(preview_image, width=110, caption=os.path.basename(sample_path))
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='side-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Notes</div>", unsafe_allow_html=True)
        st.write("- Confirm pneumonia flag with radiology team.")
        st.write("- Share Grad-CAM findings with clinician.")
        st.write("- Review scan before final reporting.")
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
with st.expander("About this dashboard"):
    st.write(
        "This healthcare-grade demo dashboard presents chest X-ray analysis with an explainable AI workflow. "
        "It combines model output, confidence, and Grad-CAM visualizations for clinical review."
    )
    st.write(
        "The model is intended for research and demonstration only and should not be used as a diagnostic device."
    )

with st.expander("Instructions"):
    st.write("- Upload a chest X-ray image to analyze.")
    st.write("- The app resizes the image to 224×224 and normalizes pixel values.")
    st.write("- Grad-CAM highlights model attention regions.")
    st.write("- Use the example scan button for a quick preview.")
