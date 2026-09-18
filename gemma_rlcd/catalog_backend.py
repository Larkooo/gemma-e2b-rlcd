"""Reference alternative: complete schema in one prefix, short parallel selectors."""

import json
from collections.abc import Sequence
from dataclasses import replace

from .cached_backend import CachedMLXBackend, PreparedState
from .core import ScoringRequest, State


class CatalogMLXBackend(CachedMLXBackend):
    """All fields share a schema-conditioned prefix, so field sets can interact.

    Use this as an experimental comparison, not as a numerically equivalent
    replacement for independently conditioned questions.
    """

    def prepare(self, state: State, requests: Sequence[ScoringRequest]) -> PreparedState:
        if not requests:
            raise ValueError("At least one scoring request is required")
        if len(requests) > self.branch_batch_size:
            raise ValueError("Catalog mode requires all fields to fit one branch batch")
        catalog = []
        selectors = []
        for i, request in enumerate(requests):
            if request.instructions is None or request.criteria is None:
                raise ValueError("Catalog mode requires structured instructions and criteria")
            catalog.append(
                {
                    "field": str(i),
                    "question": request.instructions,
                    "options": [
                        {"code": code, "name": name, "meaning": description}
                        for code, (name, description) in zip(
                            request.symbols, request.criteria, strict=True
                        )
                    ],
                }
            )
            selectors.append(replace(request, prompt=f"Evaluate field {i}. Answer code:"))
        # The parent renders multimodal state first. Add the catalog at the branch
        # marker, after every media span, so all selectors share its computation.
        self._catalog = json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
        try:
            return super().prepare(state, selectors)
        finally:
            self._catalog = None

    def prefix_task(self) -> str:
        return (
            "\n\nDecision catalog (instructions and all allowed answers):\n"
            + self._catalog
            + "\n\nDecision task:\n"
        )
