"""Runtime configuration for PaperForge."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- canonical schema
# Every enricher maps its raw payload into these fields. Nothing else survives
# into the research dataset, which is what keeps exports comparable across runs.
CANONICAL_FIELDS = [
    "title",
    "doi",
    "arxiv_id",
    "openalex_id",
    "authors",
    "affiliations",
    "year",
    "venue",
    "venue_type",
    "publisher",
    "abstract",
    "citation_count",
    "reference_count",
    "fields_of_study",
    "keywords",
    "open_access",
    "oa_url",
    "license",
    "funders",
    "language",
    "retracted",
]

# Fields the LLM extractor derives from the abstract/full text rather than
# reading off a structured record.
DERIVED_FIELDS = [
    "research_problem",
    "method_family",
    "datasets_used",
    "evaluation_metrics",
    "key_contribution",
    "limitations_stated",
    "reproducibility_signals",
    "study_type",
]

# Source trust weights. Used by resolve.py when two sources disagree on a field.
# Higher wins; ties fall back to recency of the fetch.
SOURCE_TRUST = {
    "crossref": 0.95,       # publisher-deposited, authoritative for DOI metadata
    "openalex": 0.90,       # broad + well normalised, occasionally stale
    "semantic_scholar": 0.85,
    "arxiv": 0.80,          # authoritative for preprints only
    "unpaywall": 0.90,      # authoritative for OA status, nothing else
    "llm": 0.55,            # never allowed to overwrite a structured field
}

# Per-field overrides: some sources are authoritative for a narrow slice.
FIELD_TRUST_OVERRIDES = {
    ("unpaywall", "open_access"): 0.99,
    ("unpaywall", "oa_url"): 0.99,
    ("unpaywall", "license"): 0.97,
    ("arxiv", "abstract"): 0.92,
    ("crossref", "retracted"): 0.99,
    ("openalex", "citation_count"): 0.93,
    ("semantic_scholar", "fields_of_study"): 0.90,
}


@dataclass
class RunConfig:
    """One pipeline run. Serialised into the run manifest for reproducibility."""

    mailto: str = os.getenv("PAPERFORGE_MAILTO", "research@example.org")
    sources: list[str] = field(
        default_factory=lambda: ["openalex", "crossref", "arxiv", "semantic_scholar", "unpaywall"]
    )
    concurrency: int = 8
    timeout_s: float = 20.0
    max_retries: int = 3
    use_cache: bool = True
    cache_ttl_days: int = 30

    # semantic layer
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dedup_threshold: float = 0.93
    cluster_threshold: float = 0.72

    # llm layer
    llm_enabled: bool = True
    llm_model: str = "claude-sonnet-4-6"
    llm_max_chars: int = 6000
    llm_batch_size: int = 4

    # review gate
    review_below: float = 0.70  # rows under this confidence are flagged for a human

    def to_dict(self) -> dict:
        return asdict(self)


USER_AGENT_TMPL = "PaperForge/0.2 (+https://github.com/; mailto:{mailto})"
