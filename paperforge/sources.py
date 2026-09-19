"""Async adapters for the public bibliographic APIs.

Each adapter does three things and nothing else:
  1. resolve an input reference to a source-native lookup
  2. fetch (through the cache, with backoff)
  3. map the payload into Claims against the canonical schema

Adding a source means writing one class and registering it. The pipeline never
learns anything source-specific.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from .cache import ResponseCache
from .config import SOURCE_TRUST, FIELD_TRUST_OVERRIDES, USER_AGENT_TMPL
from .models import Claim, PaperRecord

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def classify_ref(ref: str) -> tuple[str, str]:
    """Return (ref_type, normalised_value) for a free-form input reference."""
    ref = (ref or "").strip()
    if m := DOI_RE.search(ref):
        return "doi", m.group(0).lower().rstrip(".")
    if "arxiv" in ref.lower() and (m := ARXIV_RE.search(ref)):
        return "arxiv", m.group(1)
    if ARXIV_RE.fullmatch(ref):
        return "arxiv", ref
    if ref.upper().startswith("W") and ref[1:].isdigit():
        return "openalex", ref.upper()
    return "title", ref


def _trust(source: str, field_name: str) -> float:
    return FIELD_TRUST_OVERRIDES.get((source, field_name), SOURCE_TRUST.get(source, 0.5))


class Source:
    name = "base"
    base_url = ""

    def __init__(self, client: httpx.AsyncClient, cache: ResponseCache | None, mailto: str):
        self.client = client
        self.cache = cache
        self.mailto = mailto

    # -------------------------------------------------------------- transport
    async def _get(self, url: str, params: dict | None = None, as_xml: bool = False) -> Any:
        key = hashlib.sha1(f"{self.name}|{url}|{sorted((params or {}).items())}".encode()).hexdigest()
        if self.cache:
            if (hit := self.cache.get(key)) is not None:
                return hit.get("_xml") if as_xml else hit

        delay = 1.0
        for attempt in range(3):
            try:
                r = await self.client.get(
                    url,
                    params=params,
                    headers={"User-Agent": USER_AGENT_TMPL.format(mailto=self.mailto)},
                )
                if r.status_code == 429:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                payload = {"_xml": r.text} if as_xml else r.json()
                if self.cache:
                    self.cache.put(key, self.name, payload, r.status_code)
                return payload.get("_xml") if as_xml else payload
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError):
                if attempt == 2:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        return None

    # ------------------------------------------------------------- interface
    async def enrich(self, rec: PaperRecord) -> None:
        raise NotImplementedError

    def claim(self, rec: PaperRecord, name: str, value: Any, url: str, path: str) -> None:
        rec.add_claim(
            Claim(
                field_name=name,
                value=value,
                source=self.name,
                confidence=_trust(self.name, name),
                url=url,
                raw_path=path,
            )
        )


# --------------------------------------------------------------------- OpenAlex
class OpenAlex(Source):
    name = "openalex"
    base_url = "https://api.openalex.org/works"

    async def enrich(self, rec: PaperRecord) -> None:
        if rec.ref_type == "doi":
            data = await self._get(f"{self.base_url}/https://doi.org/{rec.input_ref_value}")
        elif rec.ref_type == "openalex":
            data = await self._get(f"{self.base_url}/{rec.input_ref_value}")
        else:
            res = await self._get(
                self.base_url,
                {"search": rec.input_ref_value, "per-page": 1, "mailto": self.mailto},
            )
            data = (res or {}).get("results", [None])[0] if res else None
        if not data:
            return

        url = data.get("id", self.base_url)
        c = lambda n, v, p: self.claim(rec, n, v, url, p)  # noqa: E731

        c("openalex_id", (data.get("id") or "").rsplit("/", 1)[-1], "/id")
        c("title", data.get("title"), "/title")
        c("doi", (data.get("doi") or "").replace("https://doi.org/", "") or None, "/doi")
        c("year", data.get("publication_year"), "/publication_year")
        c("citation_count", data.get("cited_by_count"), "/cited_by_count")
        c("reference_count", len(data.get("referenced_works") or []), "/referenced_works")
        c("language", data.get("language"), "/language")
        c("retracted", data.get("is_retracted"), "/is_retracted")

        authorships = data.get("authorships") or []
        c("authors", [a["author"]["display_name"] for a in authorships if a.get("author")], "/authorships")
        affs = sorted({i["display_name"] for a in authorships for i in (a.get("institutions") or [])})
        c("affiliations", affs, "/authorships/institutions")

        loc = (data.get("primary_location") or {}).get("source") or {}
        c("venue", loc.get("display_name"), "/primary_location/source")
        c("venue_type", loc.get("type"), "/primary_location/source/type")
        c("publisher", loc.get("host_organization_name"), "/primary_location/source/host_organization_name")

        oa = data.get("open_access") or {}
        c("open_access", oa.get("is_oa"), "/open_access/is_oa")
        c("oa_url", oa.get("oa_url"), "/open_access/oa_url")

        c("fields_of_study", [t["display_name"] for t in (data.get("topics") or [])[:5]], "/topics")
        c("keywords", [k["display_name"] for k in (data.get("keywords") or [])[:10]], "/keywords")
        c("funders", sorted({g.get("funder_display_name") for g in (data.get("grants") or []) if g.get("funder_display_name")}), "/grants")

        if inv := data.get("abstract_inverted_index"):
            c("abstract", _deinvert(inv), "/abstract_inverted_index")


def _deinvert(index: dict[str, list[int]]) -> str:
    positions: dict[int, str] = {}
    for word, idxs in index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


# --------------------------------------------------------------------- Crossref
class Crossref(Source):
    name = "crossref"
    base_url = "https://api.crossref.org/works"

    async def enrich(self, rec: PaperRecord) -> None:
        doi = rec.get("doi") or (rec.input_ref_value if rec.ref_type == "doi" else None)
        if doi:
            res = await self._get(f"{self.base_url}/{doi}", {"mailto": self.mailto})
            data = (res or {}).get("message")
        else:
            res = await self._get(
                self.base_url,
                {"query.bibliographic": rec.input_ref_value, "rows": 1, "mailto": self.mailto},
            )
            items = ((res or {}).get("message") or {}).get("items") or []
            data = items[0] if items else None
        if not data:
            return

        url = data.get("URL", self.base_url)
        c = lambda n, v, p: self.claim(rec, n, v, url, p)  # noqa: E731

        c("doi", (data.get("DOI") or "").lower(), "/DOI")
        c("title", (data.get("title") or [None])[0], "/title/0")
        c("publisher", data.get("publisher"), "/publisher")
        c("venue", (data.get("container-title") or [None])[0], "/container-title/0")
        c("venue_type", data.get("type"), "/type")
        c("reference_count", data.get("reference-count"), "/reference-count")
        c("language", data.get("language"), "/language")
        c("retracted", bool(data.get("update-to")), "/update-to")

        parts = (data.get("issued") or {}).get("date-parts") or [[None]]
        c("year", parts[0][0], "/issued/date-parts/0/0")

        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")])) or a.get("name")
            for a in (data.get("author") or [])
        ]
        c("authors", [a for a in authors if a], "/author")
        c("affiliations", sorted({af["name"] for a in (data.get("author") or []) for af in (a.get("affiliation") or []) if af.get("name")}), "/author/affiliation")
        c("funders", sorted({f["name"] for f in (data.get("funder") or []) if f.get("name")}), "/funder")
        c("license", ((data.get("license") or [{}])[0]).get("URL"), "/license/0/URL")
        if abs_raw := data.get("abstract"):
            c("abstract", re.sub(r"<[^>]+>", " ", abs_raw).strip(), "/abstract")


# ----------------------------------------------------------------------- arXiv
class ArXiv(Source):
    name = "arxiv"
    base_url = "http://export.arxiv.org/api/query"
    NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    async def enrich(self, rec: PaperRecord) -> None:
        if rec.ref_type == "arxiv":
            params = {"id_list": rec.input_ref_value, "max_results": 1}
        else:
            title = rec.get("title") or rec.input_ref_value
            params = {"search_query": f'ti:"{title}"', "max_results": 1}
        xml = await self._get(self.base_url, params, as_xml=True)
        if not xml:
            return
        try:
            entry = ET.fromstring(xml).find("a:entry", self.NS)
        except ET.ParseError:
            return
        if entry is None:
            return

        def text(tag: str) -> str | None:
            el = entry.find(tag, self.NS)
            return " ".join(el.text.split()) if el is not None and el.text else None

        url = text("a:id") or self.base_url
        c = lambda n, v, p: self.claim(rec, n, v, url, p)  # noqa: E731

        c("title", text("a:title"), "/entry/title")
        c("abstract", text("a:summary"), "/entry/summary")
        c("authors", [e.text for e in entry.findall("a:author/a:name", self.NS) if e.text], "/entry/author")
        if (pub := text("a:published")):
            c("year", int(pub[:4]), "/entry/published")
        if url and (m := ARXIV_RE.search(url)):
            c("arxiv_id", m.group(1), "/entry/id")
        if (doi := text("arxiv:doi")):
            c("doi", doi.lower(), "/entry/doi")
        c("fields_of_study", [e.get("term") for e in entry.findall("a:category", self.NS)], "/entry/category")
        c("venue_type", "preprint", "/entry")
        c("open_access", True, "/entry")


# -------------------------------------------------------------- Semantic Scholar
class SemanticScholar(Source):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "title,abstract,year,venue,publicationTypes,citationCount,referenceCount,fieldsOfStudy,authors.name,externalIds,openAccessPdf"

    async def enrich(self, rec: PaperRecord) -> None:
        doi = rec.get("doi")
        arxiv = rec.get("arxiv_id")
        if doi:
            data = await self._get(f"{self.base_url}/DOI:{doi}", {"fields": self.FIELDS})
        elif arxiv:
            data = await self._get(f"{self.base_url}/arXiv:{arxiv}", {"fields": self.FIELDS})
        else:
            res = await self._get(
                f"{self.base_url}/search",
                {"query": rec.input_ref_value, "limit": 1, "fields": self.FIELDS},
            )
            hits = (res or {}).get("data") or []
            data = hits[0] if hits else None
        if not data:
            return

        pid = data.get("paperId", "")
        url = f"https://www.semanticscholar.org/paper/{pid}"
        c = lambda n, v, p: self.claim(rec, n, v, url, p)  # noqa: E731

        c("title", data.get("title"), "/title")
        c("abstract", data.get("abstract"), "/abstract")
        c("year", data.get("year"), "/year")
        c("venue", data.get("venue"), "/venue")
        c("venue_type", (data.get("publicationTypes") or [None])[0], "/publicationTypes/0")
        c("citation_count", data.get("citationCount"), "/citationCount")
        c("reference_count", data.get("referenceCount"), "/referenceCount")
        c("fields_of_study", data.get("fieldsOfStudy"), "/fieldsOfStudy")
        c("authors", [a["name"] for a in (data.get("authors") or []) if a.get("name")], "/authors")
        ext = data.get("externalIds") or {}
        c("doi", (ext.get("DOI") or "").lower() or None, "/externalIds/DOI")
        c("arxiv_id", ext.get("ArXiv"), "/externalIds/ArXiv")
        if pdf := (data.get("openAccessPdf") or {}).get("url"):
            c("oa_url", pdf, "/openAccessPdf/url")
            c("open_access", True, "/openAccessPdf")


# ------------------------------------------------------------------- Unpaywall
class Unpaywall(Source):
    name = "unpaywall"
    base_url = "https://api.unpaywall.org/v2"

    async def enrich(self, rec: PaperRecord) -> None:
        doi = rec.get("doi")
        if not doi:
            return
        data = await self._get(f"{self.base_url}/{doi}", {"email": self.mailto})
        if not data:
            return
        url = f"https://api.unpaywall.org/v2/{doi}"
        c = lambda n, v, p: self.claim(rec, n, v, url, p)  # noqa: E731

        c("open_access", data.get("is_oa"), "/is_oa")
        best = data.get("best_oa_location") or {}
        c("oa_url", best.get("url_for_pdf") or best.get("url"), "/best_oa_location/url")
        c("license", best.get("license"), "/best_oa_location/license")
        c("publisher", data.get("publisher"), "/publisher")
        c("venue", data.get("journal_name"), "/journal_name")
        c("year", data.get("year"), "/year")


REGISTRY: dict[str, type[Source]] = {
    OpenAlex.name: OpenAlex,
    Crossref.name: Crossref,
    ArXiv.name: ArXiv,
    SemanticScholar.name: SemanticScholar,
    Unpaywall.name: Unpaywall,
}

# Sources that need a DOI/title already resolved run in a second wave.
WAVE_ONE = ["openalex", "crossref", "arxiv"]
WAVE_TWO = ["semantic_scholar", "unpaywall"]
