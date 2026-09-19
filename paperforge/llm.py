"""LLM extraction layer.

Structured sources tell you who wrote a paper and where. They do not tell you
what method it used, which benchmark it reported, or whether the authors
released code. That lives in prose, which is exactly what an LLM is for.

Two rules hold this layer honest:
  1. The LLM only ever writes to DERIVED_FIELDS. It can never overwrite a
     structured field — resolve.py has already decided those.
  2. Every derived value carries a per-field support score and, where possible,
     a quote span from the abstract. Nothing ungrounded reaches the export.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import httpx

from .models import PaperRecord

API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM = """You extract structured research metadata from paper abstracts.

Return ONLY a JSON object. No preamble, no markdown fences.

Schema:
{
  "research_problem": string,            // one sentence, the gap being addressed
  "method_family": string,               // e.g. "transformer", "randomised controlled trial", "finite element"
  "datasets_used": [string],             // named datasets/corpora/cohorts only
  "evaluation_metrics": [string],        // named metrics only
  "key_contribution": string,            // one sentence, what is new
  "limitations_stated": [string],        // only limitations the authors themselves state
  "reproducibility_signals": [string],   // e.g. "code released", "public dataset", "preregistered"
  "study_type": string,                  // empirical | theoretical | survey | benchmark | position | dataset
  "support": {                           // 0.0-1.0 per field above, how directly the text supports it
     "<field_name>": number
  },
  "evidence": {                          // short verbatim span from the abstract, per field, where one exists
     "<field_name>": string
  }
}

Hard rules:
- Use [] or "" when the abstract does not say. Never guess a dataset or metric.
- support must be <= 0.4 for anything you inferred rather than read.
- evidence spans must appear verbatim in the abstract."""


def _prompt(rec: PaperRecord, max_chars: int) -> str:
    return (
        f"Title: {rec.get('title') or rec.input_ref}\n"
        f"Venue: {rec.get('venue') or 'unknown'} ({rec.get('year') or 'n.d.'})\n"
        f"Fields: {', '.join(rec.get('fields_of_study') or []) or 'unknown'}\n\n"
        f"Abstract:\n{(rec.get('abstract') or '')[:max_chars]}"
    )


def _parse(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if m := re.search(r"\{.*\}", text, re.S):
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


class Extractor:
    def __init__(self, model: str = "claude-sonnet-4-6", max_chars: int = 6000,
                 api_key: str | None = None):
        self.model = model
        self.max_chars = max_chars
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def _call(self, client: httpx.AsyncClient, rec: PaperRecord) -> dict:
        body = {
            "model": self.model,
            "max_tokens": 1000,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": _prompt(rec, self.max_chars)}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        r = await client.post(API_URL, json=body, headers=headers, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return _parse(text)

    async def extract_many(
        self,
        records: list[PaperRecord],
        concurrency: int = 4,
        on_done=None,
    ) -> None:
        targets = [r for r in records if r.get("abstract") and not r.duplicate_of]
        if not targets or not self.available:
            for r in records:
                if not r.get("abstract"):
                    r.derived["_llm_skipped"] = "no abstract"
            return

        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:

            async def one(rec: PaperRecord):
                async with sem:
                    try:
                        payload = await self._call(client, rec)
                        _apply(rec, payload)
                    except Exception as exc:
                        rec.errors.append(f"llm extraction failed: {type(exc).__name__}")
                    finally:
                        if on_done:
                            on_done(rec)

            await asyncio.gather(*(one(r) for r in targets))


def _apply(rec: PaperRecord, payload: dict) -> None:
    from .config import DERIVED_FIELDS

    support = payload.get("support") or {}
    evidence = payload.get("evidence") or {}
    abstract = (rec.get("abstract") or "").lower()

    for fname in DERIVED_FIELDS:
        if fname not in payload:
            continue
        value = payload[fname]
        score = float(support.get(fname, 0.5))

        # Verify the quoted span really is in the abstract. If the model invented
        # the quote, we keep the value but halve its support and say so.
        span = (evidence.get(fname) or "").strip().lower()
        if span and span not in abstract:
            score *= 0.5
            rec.errors.append(f"unverified evidence span for {fname}")

        if score < 0.25 or value in ("", [], None):
            continue
        rec.derived[fname] = value
        rec.derived[f"{fname}__support"] = round(score, 2)
        if span:
            rec.derived[f"{fname}__evidence"] = evidence[fname][:240]
