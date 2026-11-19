import pytest
from pydantic import ValidationError

from evallab.config import settings
from evallab.golden import GoldenItem, dataset_card, load_golden, validate_sources


def test_golden_file_is_valid_and_balanced():
    items = load_golden(settings.golden_path)
    assert len(items) >= 30
    assert validate_sources(items, settings.corpus_dir) == []
    card = dataset_card(items)
    # every category must be represented, or a whole failure mode goes untested
    assert set(card["by_category"]) == {"single_hop", "multi_hop", "numeric_reasoning", "unanswerable"}
    assert card["by_category"]["unanswerable"] >= 3


def test_schema_rejects_bad_items():
    with pytest.raises(ValidationError):
        GoldenItem(id="x1", question="What?", reference="r", category="single_hop")
    with pytest.raises(ValidationError):
        GoldenItem(id="g999", question="Is this a statement", reference="r", category="single_hop")
    with pytest.raises(ValidationError):
        GoldenItem(id="g999", question="Valid question here?", reference="r", category="trivia")
