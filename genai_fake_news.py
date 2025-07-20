import argparse
import logging
import re
import sys
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.data_utils import load_data, shuffle_data
from src.explain_utils import lime_explanation, shap_explanation
from src.model_utils import get_classifier, get_device_info, predict_proba

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Fake News Detection & Explainability")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--batch_size", type=int, help="Batch size for inference")
    parser.add_argument("--model", type=str, help="Model name or path")
    return parser.parse_args()


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent code injection and strip unwanted characters."""
    text = re.sub(r"[^\x20-\x7E\n\r]", "", text)
    text = text.strip()
    return text


def main():
    args = parse_args()
    config = load_config(args.config)
    # Override config with CLI args if provided
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.model:
        config["finetuned_model_dir"] = args.model

    # Data loading
    logger.info("Loading data...")
    df = load_data(config["data_dir"])
    df = shuffle_data(df, seed=config.get("seed", 42))
    docs = df["text"].tolist()
    labels = df["label"].tolist()

    # Model loading
    classifier = get_classifier(
        finetuned_model_dir=config["finetuned_model_dir"],
        default_model=config["default_model"],
    )
    logger.info(f"Device: {get_device_info()}")

    # Batch inference
    logger.info("Evaluating model on the entire dataset...")
    batch_size = config.get("batch_size", 32)
    preds = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Batches", unit="batch"):
        batch = docs[i : i + batch_size]
        preds.extend(classifier(batch, truncation=True))
    pred_labels = [0 if p["label"] == "NEGATIVE" else 1 for p in preds]
    df["predicted_label"] = pred_labels

    # Metrics
    acc = accuracy_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels)
    rec = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    cm = confusion_matrix(labels, pred_labels)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall: {rec:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(
        "Classification Report:\n%s",
        classification_report(labels, pred_labels, target_names=["Fake", "Real"]),
    )

    # Save predictions
    output_pred_csv = config.get("output_pred_csv", "predictions.csv")
    df.to_csv(output_pred_csv, index=False)
    logger.info(f"Predictions saved to {output_pred_csv}")

    # User input for explanation
    print("\n[Privacy Note] No user data is stored. All processing is local.")
    user_input = input(
        f"Enter article index (0-{len(docs)-1}) or paste custom news text for explanation "
        f"(leave blank for index 0): "
    )
    sanitized_input = sanitize_input(user_input)
    if sanitized_input.strip() == "":
        sample_idx = 0
        sample_text = docs[sample_idx]
        logger.info("Using article at index 0.")
    else:
        try:
            sample_idx = int(sanitized_input)
            if not (0 <= sample_idx < len(docs)):
                logger.warning("Index out of range. Using 0.")
                sample_idx = 0
            sample_text = docs[sample_idx]
            logger.info(f"Using article at index {sample_idx}.")
        except ValueError:
            sample_text = sanitized_input
            sample_idx = None
            if len(sample_text) < 30:
                print("[Error] Please enter at least 30 characters for a meaningful explanation.")
                sys.exit(1)
            if len(sample_text) > 3000:
                print("[Error] Text is too long. Please limit to 3000 characters.")
                sys.exit(1)
            logger.info("Using custom text for explanation.")

    # LIME explanation
    lime_path = f"{config.get('lime_explanation_prefix', 'lime_explanation_')}{sample_idx if sample_idx is not None else 'custom'}.html"
    exp = lime_explanation(
        sample_text,
        lambda texts: predict_proba(classifier, texts),
        class_names=["Fake", "Real"],
        num_features=10,
        save_path=lime_path,
    )
    logger.info(f"LIME explanation saved to {lime_path}")
    print("LIME explanation:", exp.as_list())

    # SHAP explanation
    shap_path = f"{config.get('shap_explanation_prefix', 'shap_explanation_')}{sample_idx if sample_idx is not None else 'custom'}.html"
    shap_explanation(
        sample_text,
        lambda texts: predict_proba(classifier, texts),
        tokenizer=classifier.tokenizer,
        save_path=shap_path,
    )
    logger.info(f"SHAP explanation saved to {shap_path}")

    # Print sample and prediction
    print("Sample text:")
    print(sample_text)
    if sample_idx is not None:
        print("True label:", "Fake" if labels[sample_idx] == 0 else "Real")
        print(
            "Model prediction:",
            preds[sample_idx]["label"],
            f"(confidence: {preds[sample_idx]['score']:.2f})",
        )
    else:
        pred = classifier(sample_text)[0]
        print(
            "Model prediction:", pred["label"], f"(confidence: {pred['score']:.2f})"
        )


if __name__ == "__main__":
    main()
