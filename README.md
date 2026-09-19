# PaperForge

A Streamlit pipeline for turning a loose list of paper references into a standardized,
research-ready dataset — with every value traceable back to the source that supplied it.

Give it DOIs, arXiv IDs, arXiv URLs, or plain titles. It queries five bibliographic
sources in parallel, reconciles them where they disagree, catches the duplicates that
string matching misses, pulls method and dataset details out of abstracts, and exports
to Excel, BibTeX, RIS, JSONL, and Google Sheets.

## What it does differently

Most enrichment pipelines take the first source that answers, attach a confidence score
afterwards, and move on. PaperForge inverts that:

**Every source contributes a claim; nothing is overwritten.** Each source writes `Claim`
objects — a field, a value, a trust weight, a URL, and a JSON pointer into the raw
payload. `resolve.py` then picks a winner per field. A value three sources agree on is
not treated the same as one a single source guessed, even when the strings are identical.
The evidence log is a record of what actually happened, not a reconstruction.

**Confidence falls out of the reconciliation.** Row confidence is coverage of core
fields × mean field confidence × a penalty for contested fields. Nothing is hand-tuned
after the fact.

**Per-field trust, not per-source trust.** Unpaywall is authoritative for open-access
status and worthless for citation counts. arXiv is authoritative for preprint abstracts
and wrong about publication year. `FIELD_TRUST_OVERRIDES` in `config.py` encodes those
narrow authorities, so the right source wins each field rather than one source winning
everything.

**Two-wave fetching.** Unpaywall and Semantic Scholar are far more accurate keyed on a
DOI than on a title string. Wave one runs OpenAlex, Crossref and arXiv to find an
identifier; wave two uses it. Both waves are fully async.

**Embeddings for dedup and clustering.** The same paper appears as a preprint, a
camera-ready, and a journal extension under three different titles. Cosine similarity
over title + authors + abstract catches that; title matching does not. The same index
then powers semantic search over the corpus. Falls back to TF-IDF if
`sentence-transformers` isn't installed.

**LLM extraction, held to the evidence.** Structured sources tell you who wrote a paper.
They don't tell you which benchmark it reported or whether code was released. Claude
reads the abstract for that — but it can only write to `DERIVED_FIELDS`, never overwrite
a structured field, and every derived value carries a support score plus a verbatim span.
If the quoted span isn't actually in the abstract, the support score is halved and the
row is flagged.

**Reproducible runs.** Responses are cached in SQLite and runs are hashed over inputs +
config. Replaying a run months later gives identical payloads, so a reviewer can audit
the evidence log even after OpenAlex has moved on.

## Sources

| Source | Authoritative for | Keyed on |
|---|---|---|
| Crossref | DOI metadata, publisher, retraction notices | DOI or bibliographic query |
| OpenAlex | citation counts, topics, institutions, funders | DOI, OpenAlex ID, or search |
| arXiv | preprint abstracts and categories | arXiv ID or title |
| Semantic Scholar | fields of study, citation graph | DOI or arXiv ID |
| Unpaywall | open-access status, OA location, license | DOI |

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...        # optional — abstract extraction
export PAPERFORGE_MAILTO=you@uni.edu   # gets you a faster Crossref rate-limit pool
streamlit run app.py
```

`sentence-transformers`, `gspread` and `google-auth` are optional. Without them the app
still runs — embeddings fall back to TF-IDF and the Sheets tabs go quiet.

Try `sample_references.csv`; it deliberately mixes DOIs, bare arXiv IDs, URLs, and
title-only rows so you can see the review queue and conflict detection do something.

## Outputs

| Artefact | For |
|---|---|
| `research.xlsx` | six sheets: dataset, review queue, source conflicts, clusters, evidence log, run manifest |
| `corpus.bib` | LaTeX bibliography — flagged rows excluded by default |
| `corpus.ris` | Zotero / Mendeley / EndNote |
| `corpus.jsonl` | one record per line with full provenance, for downstream model work |
| `evidence.csv` | every claim from every source, won or lost |
| Google Sheet | dataset + review queue written back as tabs |

## Structure

```
.
├── app.py                      Streamlit UI
├── requirements.txt
├── sample_references.csv
├── .streamlit/config.toml
└── paperforge/
    ├── config.py               canonical schema, trust weights, RunConfig
    ├── models.py               Claim / Resolution / PaperRecord
    ├── cache.py                SQLite response cache + run manifests
    ├── sources.py              async adapters, one class per source
    ├── resolve.py              conflict resolution + confidence + sanity checks
    ├── semantic.py             embeddings, dedup, clustering, search
    ├── llm.py                  Claude extraction, evidence-verified
    ├── exports.py              Excel / BibTeX / RIS / JSONL / evidence log
    ├── gsheets.py              Sheets import + export
    └── pipeline.py             async orchestration, waves, manifest
```

## Adding a source

Subclass `Source`, implement `enrich()`, map the payload with `self.claim(...)`, register
it in `REGISTRY`, and add it to `WAVE_ONE` or `WAVE_TWO` depending on whether it needs an
identifier. Nothing else in the pipeline learns about it.

## Known limits

- Title-only matching takes the top hit from OpenAlex or Crossref. Those rows are always
  flagged for review — treat them as suggestions, not answers.
- `person_key` collapses authors on surname + first initial, which will merge genuine
  namesakes in large corpora. ORCID resolution would fix it and isn't wired up yet.
- Cluster labels come from OpenAlex topics and keywords, so sparsely tagged corpora get
  unhelpful labels.
- Sequential clustering means adding references means re-running the semantic stage.
