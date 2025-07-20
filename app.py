import streamlit as st
import yaml
import logging
from src.model_utils import get_classifier, predict_proba, get_device_info
from src.explain_utils import lime_explanation
import numpy as np
import os
import re

st.set_page_config(page_title="Fake News Detection with GenAI & LIME", layout="wide")
st.title("📰 Fake News Detection with GenAI & LIME Explainability")

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Config loading ---
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent code injection and strip unwanted characters."""
    # Remove non-printable characters and excessive whitespace
    text = re.sub(r'[^\x20-\x7E\n\r]', '', text)
    text = text.strip()
    return text

# --- Sidebar for config ---
st.sidebar.header("App Settings")
st.sidebar.markdown("You can change these in config.yaml for advanced use.")
st.sidebar.write(f"**Model:** {config.get('finetuned_model_dir', 'distilbert-base-uncased-finetuned-sst-2-english')}")
st.sidebar.write(f"**Device:** {get_device_info()}")
st.sidebar.write(f"**Batch size:** {config.get('batch_size', 32)}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Privacy Note:** No user data is stored. All processing is local.")

# --- Instructions ---
st.markdown("""
This app classifies news articles as **Fake** or **Real** using a transformer model and explains the prediction with LIME.
- Paste or type a news article below.
- Click **Classify & Explain** to see the result and explanation.
- You can download the LIME explanation as an HTML file.
""")

# --- Load the HuggingFace pipeline ---
@st.cache_resource(show_spinner=True)
def load_classifier():
    return get_classifier(
        finetuned_model_dir=config['finetuned_model_dir'],
        default_model=config['default_model']
    )

classifier = load_classifier()

# --- Main input area ---
text = st.text_area("Paste news article here:", height=200, max_chars=3000, help="Max 3000 characters.")

if st.button("Classify & Explain"):
    sanitized_text = sanitize_input(text)
    if not sanitized_text:
        st.warning("Please enter some text.")
    elif len(sanitized_text) < 30:
        st.warning("Please enter at least 30 characters for a meaningful prediction.")
    elif len(sanitized_text) > 3000:
        st.warning("Text is too long. Please limit to 3000 characters.")
    else:
        with st.spinner("Classifying and explaining..."):
            try:
                pred = classifier(sanitized_text)[0]
                label = pred['label']
                conf = pred['score']
                label_str = "Fake" if label == "NEGATIVE" else "Real"
                color = "#ff4b4b" if label_str == "Fake" else "#4bb543"
                st.markdown(f"**Prediction:** <span style='color:{color};font-size:1.3em'><b>{label_str}</b></span> (confidence: {conf:.2f})", unsafe_allow_html=True)
                exp = lime_explanation(
                    sanitized_text,
                    lambda texts: predict_proba(classifier, texts),
                    class_names=['Fake', 'Real'],
                    num_features=10
                )
                st.markdown("**LIME Explanation (Top Features):**")
                st.write(exp.as_list())
                st.markdown("---")
                st.markdown("**Visual Explanation:**")
                st.components.v1.html(exp.as_html(), height=400, scrolling=True)
                lime_html = exp.as_html()
                st.download_button(
                    label="Download LIME Explanation as HTML",
                    data=lime_html,
                    file_name="lime_explanation.html",
                    mime="text/html"
                )
                st.success("Explanation generated successfully!")
            except Exception as e:
                logger.error(f"Error during classification or explanation: {e}")
                st.error(f"An error occurred: {e}") 