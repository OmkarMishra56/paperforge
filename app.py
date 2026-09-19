"""PaperForge — Streamlit front end."""
from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from paperforge import __version__
from paperforge.cache import ResponseCache
from paperforge.config import RunConfig
from paperforge.exports import (
    cluster_frame, conflicts_frame, evidence_frame, research_frame,
    review_frame, to_bibtex, to_excel, to_jsonl, to_ris,
)
from paperforge.gsheets import read_references, write_dataset
from paperforge.pipeline import Pipeline
from paperforge.sources import REGISTRY

st.set_page_config(page_title="PaperForge", page_icon="◈", layout="wide")

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --ink:      #161C1B;
  --muted:    #5C6A63;
  --paper:    #EEF1EC;
  --panel:    #FBFCFA;
  --line:     #D7DCD3;
  --teal:     #2B5D5E;
  --teal-dk:  #17393A;
  --moss:     #4E7A52;
  --clay:     #A24A2C;
}

html, body, [class*="css"]  { color: var(--ink); }
.stApp { background: var(--paper); }
section[data-testid="stSidebar"] { background: var(--teal-dk); }
section[data-testid="stSidebar"] * { color: #E4EAE4 !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea {
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] small { color: #AFC0B7 !important; }

.block-container { padding-top: 2.6rem; max-width: 1280px; }

/* ---------- header ---------- */
.pf-mark { display:flex; align-items:baseline; gap:.85rem; margin-bottom:.3rem; }
.pf-mark .glyph {
  font-family: "IBM Plex Mono", monospace; font-size: 1.05rem;
  color: var(--teal); border: 1px solid var(--teal); border-radius: 3px;
  padding: .05rem .4rem; letter-spacing: .02em;
}
.pf-mark h1 {
  font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
  font-size: 2.15rem; margin: 0; letter-spacing: -.01em; color: var(--ink);
}
.pf-mark .ver {
  font-family: "IBM Plex Mono", monospace; font-size: .72rem; color: var(--muted);
}
.pf-sub {
  font-family: "IBM Plex Sans", sans-serif; color: var(--muted);
  margin: .3rem 0 0 0; max-width: 66ch; line-height: 1.6; font-size: .96rem;
}
.pf-rule { border: none; border-top: 1px solid var(--line); margin: 1.4rem 0 1.6rem 0; }

/* ---------- section labels ---------- */
.pf-label {
  font-family: "IBM Plex Mono", monospace; font-size: .72rem; color: var(--teal);
  letter-spacing: .03em; margin: 0 0 .5rem 0; border-left: 2px solid var(--teal);
  padding-left: .5rem;
}

/* ---------- ledger stat strip ---------- */
.pf-ledger { display: flex; border: 1px solid var(--line); border-radius: 4px;
  overflow: hidden; background: var(--panel); margin: .2rem 0 1.6rem 0; }
.pf-cell { flex: 1; padding: .85rem 1rem; border-right: 1px solid var(--line); }
.pf-cell:last-child { border-right: none; }
.pf-cell .v { font-family: "IBM Plex Mono", monospace; font-size: 1.5rem;
  font-weight: 500; color: var(--ink); line-height: 1.1; }
.pf-cell .k { font-family: "IBM Plex Sans", sans-serif; font-size: .72rem;
  color: var(--muted); margin-top: .25rem; }
.pf-cell.flag .v { color: var(--clay); }
.pf-cell.ok .v { color: var(--moss); }

/* ---------- record card ---------- */
.pf-card { background: var(--panel); border: 1px solid var(--line); border-radius: 4px;
  padding: 1.2rem 1.4rem; }
.pf-card h3 { font-family: "Source Serif 4", serif; font-weight: 600; font-size: 1.15rem;
  margin: 0 0 .3rem 0; line-height: 1.35; }
.pf-card .byline { font-family: "IBM Plex Sans", sans-serif; font-size: .82rem;
  color: var(--muted); margin-bottom: .7rem; }
.pf-card .abstract { font-family: "IBM Plex Sans", sans-serif; font-size: .88rem;
  line-height: 1.6; color: var(--ink); }

.pf-id { font-family: "IBM Plex Mono", monospace; font-size: .74rem;
  background: var(--paper); border: 1px solid var(--line); border-radius: 3px;
  padding: .1rem .4rem; color: var(--teal-dk); }

.pf-badge { display:inline-block; font-family: "IBM Plex Mono", monospace;
  font-size: .68rem; padding: .1rem .45rem; border-radius: 3px; margin-right: .3rem; }
.pf-badge.high { background: #E3EEE0; color: var(--moss); }
.pf-badge.low  { background: #F3E2D9; color: var(--clay); }

.pf-bar-wrap { background: var(--line); border-radius: 2px; height: 6px; width: 100%; }
.pf-bar { background: var(--teal); border-radius: 2px; height: 6px; }

/* ---------- widget chrome ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  font-family: "IBM Plex Sans", sans-serif; font-size: .88rem; padding: .5rem 0;
  color: var(--muted);
}
.stTabs [aria-selected="true"] { color: var(--teal-dk) !important; font-weight: 500; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--teal) !important; }

section[data-testid="stSidebar"] div.stButton > button {
  background: transparent; border: 1px solid #3E6B69; color: #E4EAE4 !important;
  font-family: "IBM Plex Sans", sans-serif; border-radius: 3px;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
  border-color: #E4EAE4; background: rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] hr { border-color: #2C4E4D; }

div.stButton > button[kind="primary"], div.stDownloadButton > button {
  background: var(--teal-dk); border: none; border-radius: 3px;
  font-family: "IBM Plex Sans", sans-serif; font-weight: 500;
}
div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button:hover {
  background: var(--teal);
}

div[data-testid="stMetricValue"] {
  font-family: "IBM Plex Mono", monospace; font-size: 1.5rem; font-weight: 500;
}
div[data-testid="stMetricLabel"] { font-family: "IBM Plex Sans", sans-serif;
  font-size: .74rem; color: var(--muted); }

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    f"""<div class="pf-mark"><span class="glyph">◈</span><h1>PaperForge</h1>
    <span class="ver">v{__version__}</span></div>
    <p class="pf-sub">Give it a pile of DOIs, arXiv IDs, or paper titles. It queries five
    bibliographic sources in parallel, reconciles where they disagree, finds the duplicates
    string matching misses, and hands back a dataset with every value traceable to the source
    that supplied it.</p>
    <hr class="pf-rule">""",
    unsafe_allow_html=True,
)

for key, default in [
    ("records", None), ("manifest", None), ("pipeline", None), ("refs", []),
]:
    st.session_state.setdefault(key, default)


# ───────────────────────────────────────────────────────────── sidebar: settings
with st.sidebar:
    st.markdown('<p class="pf-label" style="border-left-color:#4E7A52;">Run settings</p>',
               unsafe_allow_html=True)

    picked = st.multiselect(
        "Sources", list(REGISTRY), default=list(REGISTRY),
        help="Crossref and OpenAlex resolve raw references. Unpaywall and Semantic "
             "Scholar run afterwards, once a DOI exists.",
    )
    mailto = st.text_input(
        "Contact email", os.getenv("PAPERFORGE_MAILTO", "research@example.org"),
        help="Crossref and Unpaywall give a faster rate-limit pool to identified callers.",
    )
    concurrency = st.slider("Parallel requests", 1, 24, 8)

    st.divider()
    st.caption("Semantic layer")
    dedup = st.slider("Duplicate threshold", 0.80, 0.99, 0.93, 0.01)
    cluster = st.slider("Cluster threshold", 0.50, 0.95, 0.72, 0.01)

    st.divider()
    st.caption("Extraction")
    api_key = st.text_input(
        "Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Reads method, datasets, metrics and limitations out of abstracts. "
             "Leave blank to skip — everything else still runs.",
    )
    llm_on = st.toggle("Extract from abstracts", value=bool(api_key))

    st.divider()
    use_cache = st.toggle("Use response cache", value=True)
    cache = ResponseCache(ttl_days=30)
    stats = cache.stats()
    st.caption(f"{stats['entries']} cached responses")
    if st.button("Clear cache", use_container_width=True):
        cache.clear()
        st.rerun()

cfg = RunConfig(
    mailto=mailto, sources=picked, concurrency=concurrency,
    dedup_threshold=dedup, cluster_threshold=cluster,
    llm_enabled=llm_on, use_cache=use_cache,
)


# ─────────────────────────────────────────────────────────────────────── input
st.markdown('<p class="pf-label">References</p>', unsafe_allow_html=True)
tab_paste, tab_file, tab_sheet = st.tabs(["Paste", "Upload", "Google Sheet"])
refs: list[str] = []

with tab_paste:
    raw = st.text_area(
        "One reference per line", height=170,
        placeholder="10.1038/s41586-021-03819-2\n1706.03762\nAttention is all you need\nhttps://arxiv.org/abs/2005.14165",
    )
    if raw.strip():
        refs = [line.strip() for line in raw.splitlines() if line.strip()]

with tab_file:
    up = st.file_uploader("CSV, Excel, or a plain text list", type=["csv", "xlsx", "xls", "txt"])
    if up:
        if up.name.endswith(".txt"):
            refs = [l.strip() for l in up.read().decode("utf-8", "ignore").splitlines() if l.strip()]
        else:
            df_in = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            col = st.selectbox("Reference column", df_in.columns)
            refs = [str(v).strip() for v in df_in[col].dropna() if str(v).strip()]
            st.dataframe(df_in.head(5), use_container_width=True, hide_index=True)

with tab_sheet:
    sa = st.text_area("Service account JSON", height=90,
                      placeholder='{"type": "service_account", ...}')
    sheet_url = st.text_input("Sheet URL")
    ws_name = st.text_input("Worksheet name", placeholder="leave blank for the first tab")
    if st.button("Load from sheet") and sa and sheet_url:
        try:
            refs = read_references(sa, sheet_url, ws_name or None)
            st.session_state.refs = refs
            st.success(f"Loaded {len(refs)} references")
        except Exception as exc:
            st.error(f"Could not read the sheet: {exc}")
    refs = refs or st.session_state.refs

if refs:
    st.session_state.refs = refs
refs = st.session_state.refs


# ───────────────────────────────────────────────────────────────────── the run
col_go, col_note = st.columns([1, 4])
with col_go:
    go = st.button(f"Enrich {len(refs)} references" if refs else "Enrich",
                   type="primary", disabled=not refs, use_container_width=True)
with col_note:
    if refs:
        st.caption(f"{len(picked)} sources × {len(refs)} references, "
                   f"{concurrency} at a time")

if go:
    bar = st.progress(0.0)
    line = st.empty()

    def progress(stage: str, done: int, total: int, note: str = ""):
        pct = done / total if total else 0.0
        bar.progress(min(pct, 1.0))
        line.caption(f"**{stage}** — {done}/{total} {('· ' + note) if note else ''}")

    pipe = Pipeline(cfg, cache, api_key or None)
    t0 = time.time()
    with st.spinner("Working"):
        records = pipe.run(refs, progress)
    manifest = pipe.manifest(refs, records)
    cache.save_manifest(manifest["run_id"], manifest)

    st.session_state.update(records=records, manifest=manifest, pipeline=pipe)
    bar.empty()
    line.empty()
    st.success(f"Finished {len(records)} records in {time.time() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────────── results
records = st.session_state.records
if records:
    pipe: Pipeline = st.session_state.pipeline
    manifest = st.session_state.manifest
    s = pipe.stats

    st.markdown('<hr class="pf-rule">', unsafe_allow_html=True)

    def _cell(label: str, value, cls: str = "") -> str:
        return f'<div class="pf-cell {cls}"><div class="v">{value}</div><div class="k">{label}</div></div>'

    ledger = "".join([
        _cell("records", s.total),
        _cell("enriched", s.enriched, "ok" if s.enriched == s.total else ""),
        _cell("duplicates", s.duplicates),
        _cell("clusters", s.clusters),
        _cell("need review", s.flagged, "flag" if s.flagged else "ok"),
        _cell("elapsed", f"{s.elapsed:.1f}s"),
    ])
    st.markdown(f'<div class="pf-ledger">{ledger}</div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Dataset", "Review queue", "Conflicts", "Clusters", "Search", "Export"]
    )

    with t1:
        df = research_frame(records)
        only_clean = st.checkbox("Hide flagged rows", value=False)
        view = df[~df["needs_review"]] if only_clean else df
        st.dataframe(view, use_container_width=True, hide_index=True,
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0.0, max_value=1.0, format="%.2f")})

        pick = st.selectbox(
            "Inspect a record", options=[r.record_id for r in records],
            format_func=lambda rid: next(
                (r.get("title") or r.input_ref)[:80] for r in records if r.record_id == rid),
        )
        rec = next(r for r in records if r.record_id == pick)
        left, right = st.columns([3, 2])
        with left:
            authors = rec.get("authors") or []
            byline = (
                f"{', '.join(authors[:6])}{' et al.' if len(authors) > 6 else ''} · "
                f"{rec.get('venue') or 'venue unknown'} · {rec.get('year') or 'n.d.'}"
            )
            conf = rec.confidence
            badge_cls = "high" if conf >= 0.7 else "low"
            abstract = rec.get("abstract")
            abstract_html = (
                f'<p class="abstract">{(abstract[:1200] + ("…" if len(abstract) > 1200 else ""))}</p>'
                if abstract else ""
            )
            st.markdown(
                f"""<div class="pf-card">
                  <span class="pf-id">{rec.record_id}</span>
                  <span class="pf-badge {badge_cls}">confidence {conf:.2f}</span>
                  <h3>{rec.get('title') or rec.input_ref}</h3>
                  <div class="byline">{byline}</div>
                  <div class="pf-bar-wrap"><div class="pf-bar" style="width:{conf*100:.0f}%"></div></div>
                  {abstract_html}
                </div>""",
                unsafe_allow_html=True,
            )
            if rec.derived:
                st.markdown("**Extracted from the abstract**")
                st.dataframe(
                    pd.DataFrame(
                        [{"field": k, "value": "; ".join(v) if isinstance(v, list) else v}
                         for k, v in rec.derived.items() if not k.startswith("_")
                         and not k.endswith(("__support", "__evidence"))]
                    ),
                    use_container_width=True, hide_index=True,
                )
        with right:
            st.markdown("**Field provenance**")
            st.dataframe(
                pd.DataFrame([
                    {"field": n, "source": r.source, "conf": r.confidence,
                     "agree": r.agreement, "contested": r.contested}
                    for n, r in rec.resolved.items()
                ]),
                use_container_width=True, hide_index=True, height=420,
            )

    with t2:
        rq = review_frame(records)
        if rq.empty:
            st.success("Nothing flagged. Every record cleared the confidence threshold.")
        else:
            st.caption("Edit the verified and notes columns, then export the workbook.")
            st.data_editor(rq, use_container_width=True, hide_index=True, num_rows="fixed")

    with t3:
        cf = conflicts_frame(records)
        if cf.empty:
            st.info("No field had two sources disagree.")
        else:
            st.caption("Where sources disagreed, what won, and what lost.")
            st.dataframe(cf, use_container_width=True, hide_index=True)

    with t4:
        clf = cluster_frame(records, pipe.cluster_names)
        st.dataframe(clf, use_container_width=True, hide_index=True)
        if not clf.empty and clf["papers"].sum() > 1:
            st.bar_chart(clf.set_index("label")["papers"], height=240)

    with t5:
        q = st.text_input("Search the corpus by meaning, not keywords",
                          placeholder="methods that use contrastive pretraining on graphs")
        if q:
            for rec, score in pipe.index.search(q, records, top_k=8):
                st.markdown(f"**{rec.get('title') or rec.input_ref}**  ·  {score:.3f}")
                st.caption(
                    f"{rec.get('venue') or '—'} · {rec.get('year') or 'n.d.'} · "
                    f"{rec.get('citation_count') or 0} citations"
                )

    with t6:
        st.download_button(
            "Excel workbook", to_excel(records, manifest, pipe.cluster_names),
            file_name=f"paperforge_{manifest['run_id']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("BibTeX", to_bibtex(records), "corpus.bib",
                           use_container_width=True)
        c2.download_button("RIS", to_ris(records), "corpus.ris",
                           use_container_width=True)
        c3.download_button("JSONL", to_jsonl(records, manifest), "corpus.jsonl",
                           use_container_width=True)
        c4.download_button("Evidence log",
                           evidence_frame(records).to_csv(index=False),
                           "evidence.csv", use_container_width=True)

        st.divider()
        st.caption("Push back to Google Sheets")
        out_url = st.text_input("Destination sheet URL", key="out_sheet")
        if st.button("Write to sheet") and out_url:
            try:
                url = write_dataset(sa, out_url, {
                    "Research dataset": research_frame(records),
                    "Needs review": review_frame(records),
                    "Source conflicts": conflicts_frame(records),
                })
                st.success(f"Written to {url}")
            except Exception as exc:
                st.error(f"Write failed: {exc}")

        st.divider()
        st.caption("Run manifest — hash the inputs and config to replay this run later")
        st.json(manifest, expanded=False)
else:
    st.info("Load some references to get started. "
            "A DOI, an arXiv ID, an arXiv URL, or a plain title all work.")