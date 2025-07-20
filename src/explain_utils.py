import logging
import re
import yaml

from lime.lime_text import LimeTextExplainer
import shap

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent code injection and strip unwanted characters."""
    text = re.sub(r"[^\x20-\x7E\n\r]", "", text)
    text = text.strip()
    return text


def lime_explanation(
    text: str,
    predict_proba_func,
    class_names: list[str],
    num_features: int = 10,
    save_path: str = None,
):
    """Generate a LIME explanation for the given text and prediction function. Optionally save to HTML."""
    if len(text.strip()) < 30:
        raise ValueError("Input text is too short for a meaningful explanation (min 30 characters).")
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text, predict_proba_func, num_features=num_features
    )
    if save_path:
        exp.save_to_file(save_path)
    return exp


def shap_explanation(text: str, predict_proba_func, tokenizer, save_path: str = None):
    """Generate a SHAP explanation for the given text and prediction function. Optionally save to HTML."""
    shap_explainer = shap.Explainer(
        predict_proba_func, masker=shap.maskers.Text(tokenizer=tokenizer)
    )
    shap_values = shap_explainer([
        text
    ])
    if save_path:
        # SHAP's save_html is not available in all versions. If not, fallback to saving text plot as HTML.
        try:
            shap.save_html(save_path, shap_values)
        except AttributeError:
            # Fallback: save the text plot as HTML using matplotlib
            import matplotlib.pyplot as plt
            shap.plots.text(shap_values, display=False, show=False)
            plt.savefig(save_path)
    return shap_values
