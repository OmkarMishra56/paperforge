"""Export layer.

Five artefacts, each aimed at a different downstream consumer:
  research.xlsx   — the human-readable workbook, one sheet per view
  corpus.bib      — drops straight into a LaTeX bibliography
  corpus.ris      — Zotero / Mendeley / EndNote
  corpus.jsonl    — one record per line, for downstream model work
  evidence.csv    — every claim from every source, won or lost
"""
from __future__ import annotations

import io
import json
import re
import unicodedata

import pandas as pd

from .models import PaperRecord


def research_frame(records: list[PaperRecord]) -> pd.DataFrame:
    rows = [r.flat() for r in records]
    df = pd.DataFrame(rows)
    front = ["record_id", "title", "authors", "year", "venue", "doi", "confidence", "needs_review"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    return df[cols]


def evidence_frame(records: list[PaperRecord]) -> pd.DataFrame:
    rows = [row for r in records for row in r.evidence_rows()]
    return pd.DataFrame(rows)


def conflicts_frame(records: list[PaperRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        for name, res in r.resolved.items():
            if not res.contested:
                continue
            rows.append(
                {
                    "record_id": r.record_id,
                    "title": (r.get("title") or r.input_ref)[:80],
                    "field": name,
                    "chosen": str(res.value)[:160],
                    "chosen_source": res.source,
                    "agreement": res.agreement,
                    "rejected": " | ".join(
                        f"{c.source}={str(c.value)[:60]}" for c in res.alternatives
                    ),
                }
            )
    return pd.DataFrame(rows)


def review_frame(records: list[PaperRecord]) -> pd.DataFrame:
    flagged = [r for r in records if r.needs_review]
    if not flagged:
        return pd.DataFrame(columns=["record_id", "title", "confidence", "reason"])
    rows = []
    for r in flagged:
        reasons = list(r.errors)
        if r.confidence < 0.70:
            reasons.append(f"low confidence ({r.confidence})")
        if r.get("retracted") is True:
            reasons.append("flagged as retracted")
        rows.append(
            {
                "record_id": r.record_id,
                "title": r.get("title") or r.input_ref,
                "doi": r.get("doi"),
                "confidence": r.confidence,
                "sources_hit": "; ".join(r.sources_hit),
                "reason": "; ".join(reasons),
                "verified": "",   # researcher fills this in
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def cluster_frame(records: list[PaperRecord], names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for cid in sorted({r.cluster_id for r in records if r.cluster_id is not None}):
        members = [r for r in records if r.cluster_id == cid]
        years = [r.get("year") for r in members if isinstance(r.get("year"), int)]
        rows.append(
            {
                "cluster_id": cid,
                "label": names.get(cid, f"cluster {cid}"),
                "papers": len(members),
                "year_range": f"{min(years)}–{max(years)}" if years else "",
                "median_citations": pd.Series(
                    [r.get("citation_count") or 0 for r in members]
                ).median(),
                "open_access_share": round(
                    sum(1 for r in members if r.get("open_access")) / len(members), 2
                ),
                "example": (members[0].get("title") or members[0].input_ref)[:90],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- bibtex
def _cite_key(rec: PaperRecord) -> str:
    authors = rec.get("authors") or ["anon"]
    surname = re.sub(r"[^A-Za-z]", "", authors[0].split()[-1]).lower() or "anon"
    year = rec.get("year") or "nd"
    title = rec.get("title") or ""
    word = next((w for w in re.findall(r"[A-Za-z]{4,}", title)), "paper").lower()
    return f"{surname}{year}{word}"


_BIB_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def _bib_clean(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    for k, v in _BIB_ESCAPE.items():
        text = text.replace(k, v)
    return text


def to_bibtex(records: list[PaperRecord], include_flagged: bool = False) -> str:
    out = []
    seen = set()
    for r in records:
        if r.duplicate_of or (r.needs_review and not include_flagged):
            continue
        key = _cite_key(r)
        while key in seen:
            key += "a"
        seen.add(key)

        entry_type = "article"
        vt = (r.get("venue_type") or "").lower()
        if "preprint" in vt:
            entry_type = "misc"
        elif "conference" in vt or "proceedings" in vt:
            entry_type = "inproceedings"

        fields = {
            "title": r.get("title"),
            "author": " and ".join(r.get("authors") or []),
            "year": r.get("year"),
            "journal" if entry_type == "article" else "booktitle": r.get("venue"),
            "publisher": r.get("publisher"),
            "doi": r.get("doi"),
            "url": r.get("oa_url"),
            "note": f"arXiv:{r.get('arxiv_id')}" if r.get("arxiv_id") else None,
        }
        body = ",\n".join(
            f"  {k} = {{{_bib_clean(v)}}}" for k, v in fields.items() if v
        )
        out.append(f"@{entry_type}{{{key},\n{body}\n}}")
    return "\n\n".join(out)


# ------------------------------------------------------------------------- RIS
_RIS_TYPE = {"preprint": "UNPB", "conference": "CPAPER", "journal-article": "JOUR"}


def to_ris(records: list[PaperRecord]) -> str:
    out = []
    for r in records:
        if r.duplicate_of:
            continue
        lines = [f"TY  - {_RIS_TYPE.get((r.get('venue_type') or '').lower(), 'JOUR')}"]
        for a in r.get("authors") or []:
            lines.append(f"AU  - {a}")
        for tag, value in (
            ("TI", r.get("title")), ("PY", r.get("year")), ("JO", r.get("venue")),
            ("PB", r.get("publisher")), ("DO", r.get("doi")), ("UR", r.get("oa_url")),
            ("AB", r.get("abstract")), ("LA", r.get("language")),
        ):
            if value:
                lines.append(f"{tag}  - {value}")
        for kw in (r.get("keywords") or [])[:10]:
            lines.append(f"KW  - {kw}")
        lines.append("ER  - \n")
        out.append("\n".join(lines))
    return "\n".join(out)


# ----------------------------------------------------------------------- jsonl
def to_jsonl(records: list[PaperRecord], manifest: dict | None = None) -> str:
    lines = []
    for r in records:
        payload = {
            "record_id": r.record_id,
            "input_ref": r.input_ref,
            "metadata": {name: res.value for name, res in r.resolved.items()},
            "derived": r.derived,
            "provenance": {
                name: {"source": res.source, "confidence": res.confidence,
                       "agreement": res.agreement, "contested": res.contested}
                for name, res in r.resolved.items()
            },
            "confidence": r.confidence,
            "needs_review": r.needs_review,
            "duplicate_of": r.duplicate_of,
            "cluster_id": r.cluster_id,
            "run_id": (manifest or {}).get("run_id"),
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


# ----------------------------------------------------------------------- excel
def to_excel(records: list[PaperRecord], manifest: dict, cluster_names: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        research_frame(records).to_excel(xl, sheet_name="Research dataset", index=False)
        review_frame(records).to_excel(xl, sheet_name="Needs review", index=False)
        conflicts_frame(records).to_excel(xl, sheet_name="Source conflicts", index=False)
        cluster_frame(records, cluster_names).to_excel(xl, sheet_name="Clusters", index=False)
        evidence_frame(records).to_excel(xl, sheet_name="Evidence log", index=False)
        pd.DataFrame(
            [{"key": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
             for k, v in manifest.items()]
        ).to_excel(xl, sheet_name="Run manifest", index=False)
    return buf.getvalue()
