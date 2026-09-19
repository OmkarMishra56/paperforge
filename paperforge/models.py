"""Records, field-level provenance, and the evidence log."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Claim:
    """A single value for a single field, asserted by a single source.

    The pipeline never overwrites values in place. Every source contributes a
    Claim, and resolve.py picks a winner. That is what makes the evidence log
    honest rather than reconstructed after the fact.
    """

    field_name: str
    value: Any
    source: str
    confidence: float
    fetched_at: float = field(default_factory=time.time)
    url: str | None = None
    raw_path: str | None = None  # JSON pointer into the source payload

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Resolution:
    """The winning claim for a field, plus everything it beat."""

    field_name: str
    value: Any
    source: str
    confidence: float
    agreement: float          # share of sources that matched the winner
    contested: bool
    alternatives: list[Claim] = field(default_factory=list)


@dataclass
class PaperRecord:
    """One row through the pipeline."""

    input_ref: str                      # whatever the user gave us
    ref_type: str = "unknown"           # doi | arxiv | title | openalex
    input_ref_value: str = ""           # normalised form of input_ref
    claims: list[Claim] = field(default_factory=list)
    resolved: dict[str, Resolution] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    sources_hit: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    cluster_id: int | None = None
    embedding: list[float] | None = None

    # -------------------------------------------------------------- helpers
    def add_claim(self, claim: Claim) -> None:
        if claim.value in (None, "", [], {}):
            return
        self.claims.append(claim)

    def get(self, name: str, default: Any = None) -> Any:
        if name in self.resolved:
            return self.resolved[name].value
        return self.derived.get(name, default)

    @property
    def record_id(self) -> str:
        seed = self.get("doi") or self.get("openalex_id") or self.input_ref
        return hashlib.sha1(str(seed).lower().encode()).hexdigest()[:12]

    @property
    def confidence(self) -> float:
        """Row confidence = coverage x mean field confidence x agreement penalty."""
        if not self.resolved:
            return 0.0
        core = ["title", "authors", "year", "venue", "doi", "abstract"]
        coverage = sum(1 for f in core if self.get(f)) / len(core)
        mean_conf = sum(r.confidence for r in self.resolved.values()) / len(self.resolved)
        contested = sum(1 for r in self.resolved.values() if r.contested)
        penalty = 1.0 - min(0.3, 0.05 * contested)
        return round(coverage * mean_conf * penalty, 3)

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.70 or bool(self.errors) or self.get("retracted") is True

    def flat(self) -> dict:
        """Flatten to a single export row."""
        row: dict[str, Any] = {"record_id": self.record_id, "input_ref": self.input_ref}
        for name, res in self.resolved.items():
            v = res.value
            row[name] = "; ".join(map(str, v)) if isinstance(v, list) else v
        for name, v in self.derived.items():
            row[name] = "; ".join(map(str, v)) if isinstance(v, list) else v
        row["confidence"] = self.confidence
        row["sources_hit"] = "; ".join(self.sources_hit)
        row["contested_fields"] = "; ".join(
            n for n, r in self.resolved.items() if r.contested
        )
        row["needs_review"] = self.needs_review
        row["duplicate_of"] = self.duplicate_of
        row["cluster_id"] = self.cluster_id
        row["errors"] = "; ".join(self.errors)
        return row

    def evidence_rows(self) -> list[dict]:
        out = []
        for c in self.claims:
            res = self.resolved.get(c.field_name)
            out.append(
                {
                    "record_id": self.record_id,
                    "field": c.field_name,
                    "value": json.dumps(c.value, ensure_ascii=False)[:500],
                    "source": c.source,
                    "source_confidence": c.confidence,
                    "won": bool(res and res.source == c.source),
                    "url": c.url,
                    "raw_path": c.raw_path,
                    "fetched_at": c.fetched_at,
                }
            )
        return out
