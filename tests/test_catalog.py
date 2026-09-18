import json

import pytest

from gemma_decisions.cached_backend import CachedMLXBackend
from gemma_decisions.catalog_backend import CatalogMLXBackend
from gemma_decisions.core import ScoringRequest, State


def test_catalog_retains_complete_descriptions_and_uses_short_selectors(monkeypatch):
    backend = CatalogMLXBackend.__new__(CatalogMLXBackend)
    backend.branch_batch_size = 8
    requests = [
        ScoringRequest(
            "original",
            ("A", "B"),
            "Question\nwith detail",
            (("yes", "First\nsecond line"), ("no", "No")),
        )
    ]
    captured = {}

    def prepare(self, state, selectors):
        captured["catalog"] = json.loads(self._catalog)
        captured["selectors"] = selectors
        return "prepared"

    monkeypatch.setattr(CachedMLXBackend, "prepare", prepare)
    assert backend.prepare(State(text="evidence"), requests) == "prepared"
    assert captured["catalog"][0]["question"] == "Question\nwith detail"
    assert captured["catalog"][0]["options"][0]["meaning"] == "First\nsecond line"
    assert captured["selectors"][0].prompt == "Evaluate field 0. Answer code:"
    assert backend._catalog is None


def test_catalog_rejects_overflow_instead_of_serializing_fields():
    backend = CatalogMLXBackend.__new__(CatalogMLXBackend)
    backend.branch_batch_size = 1
    with pytest.raises(ValueError, match="one branch batch"):
        backend.prepare(State(text="evidence"), [ScoringRequest("x", ("A", "B"))] * 2)
