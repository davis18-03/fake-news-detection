import pytest

from src.explain_utils import lime_explanation
from src.model_utils import get_classifier, predict_proba


def test_lime_explanation():
    clf = get_classifier(default_model="distilbert-base-uncased-finetuned-sst-2-english")
    text = "This is a fake news article. It contains enough words to exceed thirty characters."
    exp = lime_explanation(
        text,
        lambda texts: predict_proba(clf, texts),
        class_names=["Fake", "Real"],
        num_features=5,
    )
    assert hasattr(exp, "as_list")
    assert isinstance(exp.as_list(), list)


def test_lime_explanation_short_input():
    clf = get_classifier(default_model="distilbert-base-uncased-finetuned-sst-2-english")
    short_text = "Hi."
    with pytest.raises(Exception):
        lime_explanation(
            short_text,
            lambda texts: predict_proba(clf, texts),
            class_names=["Fake", "Real"],
            num_features=5,
        )
