"""Pipeline orchestration.

Stages, in order:
  1. parse    — classify each input reference
  2. wave one — sources that can resolve from a raw reference (parallel)
  3. resolve  — first pass, so wave two has a DOI to work with
  4. wave two — sources that need an identifier (parallel)
  5. resolve  — final pass over the full claim set
  6. semantic — embed, dedup, cluster
  7. derive   — LLM extraction on survivors
  8. manifest — hash inputs + config so the run can be replayed

Waves matter: Unpaywall and Semantic Scholar are far more accurate with a DOI
than with a title string, and OpenAlex/Crossref are the cheapest way to get one.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field

import httpx

from .cache import ResponseCache
from .config import RunConfig
from .llm import Extractor
from .models import PaperRecord
from .resolve import resolve_record
from .semantic import EmbeddingIndex, cluster_labels
from .sources import REGISTRY, WAVE_ONE, WAVE_TWO, classify_ref


@dataclass
class RunStats:
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    total: int = 0
    enriched: int = 0
    duplicates: int = 0
    clusters: int = 0
    flagged: int = 0
    llm_calls: int = 0
    source_hits: dict = field(default_factory=dict)
    source_errors: dict = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


class Pipeline:
    def __init__(self, cfg: RunConfig, cache: ResponseCache | None = None,
                 api_key: str | None = None):
        self.cfg = cfg
        self.cache = cache if cfg.use_cache else None
        self.stats = RunStats()
        self.extractor = Extractor(cfg.llm_model, cfg.llm_max_chars, api_key)
        self.index = EmbeddingIndex(cfg.embed_model)
        self.cluster_names: dict[int, str] = {}

    # ------------------------------------------------------------------ public
    def run(self, refs: list[str], progress=None) -> list[PaperRecord]:
        """Blocking entry point — Streamlit calls this."""
        return asyncio.run(self.run_async(refs, progress))

    async def run_async(self, refs: list[str], progress=None) -> list[PaperRecord]:
        def tick(stage: str, done: int, total: int, note: str = ""):
            if progress:
                progress(stage, done, total, note)

        records = []
        for ref in refs:
            ref = (ref or "").strip()
            if not ref:
                continue
            rtype, value = classify_ref(ref)
            records.append(PaperRecord(input_ref=ref, ref_type=rtype, input_ref_value=value))
        self.stats.total = len(records)
        tick("parse", len(records), len(records), f"{len(records)} references")

        limits = httpx.Limits(max_connections=self.cfg.concurrency * 2)
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s, limits=limits,
                                     follow_redirects=True) as client:
            wave_one = [s for s in WAVE_ONE if s in self.cfg.sources]
            wave_two = [s for s in WAVE_TWO if s in self.cfg.sources]

            await self._wave(client, records, wave_one, "enrich · primary", tick)
            for r in records:
                resolve_record(r)
            await self._wave(client, records, wave_two, "enrich · identifier-keyed", tick)

        for r in records:
            r.resolved.clear()
            r.errors.clear()
            resolve_record(r)
        self.stats.enriched = sum(1 for r in records if r.resolved)
        tick("resolve", len(records), len(records), "claims reconciled")

        # --- semantic layer
        if records:
            tick("embed", 0, len(records), f"backend: {self.index.backend}")
            self.index.fit(records)
            self.stats.duplicates = self.index.mark_duplicates(records, self.cfg.dedup_threshold)
            self.stats.clusters = self.index.cluster(records, self.cfg.cluster_threshold)
            self.cluster_names = cluster_labels(records)
            tick("embed", len(records), len(records),
                 f"{self.stats.duplicates} duplicates, {self.stats.clusters} clusters")

        # --- llm layer
        if self.cfg.llm_enabled and self.extractor.available:
            targets = [r for r in records if r.get("abstract") and not r.duplicate_of]
            done = {"n": 0}

            def on_done(_rec):
                done["n"] += 1
                self.stats.llm_calls += 1
                tick("derive", done["n"], len(targets), "extracting method & evidence")

            tick("derive", 0, len(targets), "extracting method & evidence")
            await self.extractor.extract_many(records, self.cfg.llm_batch_size, on_done)
        elif self.cfg.llm_enabled:
            tick("derive", 0, 0, "skipped — no ANTHROPIC_API_KEY set")

        self.stats.flagged = sum(1 for r in records if r.needs_review)
        self.stats.finished_at = time.time()
        return records

    # ------------------------------------------------------------------ waves
    async def _wave(self, client, records, source_names, label, tick):
        if not source_names:
            return
        sem = asyncio.Semaphore(self.cfg.concurrency)
        tasks = []
        total = len(records) * len(source_names)
        done = {"n": 0}

        async def one(rec: PaperRecord, name: str):
            async with sem:
                src = REGISTRY[name](client, self.cache, self.cfg.mailto)
                try:
                    before = len(rec.claims)
                    await src.enrich(rec)
                    if len(rec.claims) > before:
                        rec.sources_hit.append(name)
                        self.stats.source_hits[name] = self.stats.source_hits.get(name, 0) + 1
                except Exception as exc:
                    msg = f"{name}: {type(exc).__name__}"
                    rec.errors.append(msg)
                    self.stats.source_errors[name] = self.stats.source_errors.get(name, 0) + 1
                finally:
                    done["n"] += 1
                    tick(label, done["n"], total, name)

        for rec in records:
            for name in source_names:
                tasks.append(one(rec, name))
        await asyncio.gather(*tasks)

    # --------------------------------------------------------------- manifest
    def manifest(self, refs: list[str], records: list[PaperRecord]) -> dict:
        input_hash = hashlib.sha256("\n".join(sorted(refs)).encode()).hexdigest()[:16]
        cfg_hash = hashlib.sha256(
            json.dumps(self.cfg.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]
        return {
            "run_id": f"{input_hash}-{cfg_hash}",
            "input_hash": input_hash,
            "config_hash": cfg_hash,
            "config": self.cfg.to_dict(),
            "embedding_backend": self.index.backend,
            "record_ids": [r.record_id for r in records],
            "stats": {
                "total": self.stats.total,
                "enriched": self.stats.enriched,
                "duplicates": self.stats.duplicates,
                "clusters": self.stats.clusters,
                "flagged": self.stats.flagged,
                "llm_calls": self.stats.llm_calls,
                "elapsed_s": round(self.stats.elapsed, 2),
                "source_hits": self.stats.source_hits,
                "source_errors": self.stats.source_errors,
            },
            "created_at": time.time(),
        }
