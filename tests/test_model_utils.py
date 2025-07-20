import numpy as np

from src.model_utils import get_classifier, predict_proba


def test_get_classifier():
    clf = get_classifier(default_model="distilbert-base-uncased-finetuned-sst-2-english")
    assert hasattr(clf, "__call__")
    assert hasattr(clf, "model")


def test_predict_proba():
    clf = get_classifier(default_model="distilbert-base-uncased-finetuned-sst-2-english")
    texts = ["This is a fake news article.", "This is a real news article."]
    probs = predict_proba(clf, texts)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-3)
