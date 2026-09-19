"""Turn competing Claims into one resolved value per field.

SkyTrack-style pipelines usually take the first source that answers and attach a
score afterwards. Here the score falls out of the resolution itself: a field that
three sources agree on is not the same as a field one source guessed, even when
the winning string is identical.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

from .models import Claim, PaperRecord, Resolution

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

LIST_FIELDS = {"authors", "affiliations", "fields_of_study", "keywords", "funders"}
NUMERIC_FIELDS = {"year", "citation_count", "reference_count"}


def normalise(value: Any) -> Any:
    """Comparison key. Two claims 'agree' if their normal forms match."""
    if isinstance(value, str):
        s = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()
    if isinstance(value, list):
        return tuple(sorted(normalise(v) for v in value if v))
    return value


def person_key(name: str) -> str:
    """Match 'A Vaswani' to 'Ashish Vaswani'.

    Sources vary wildly on whether they abbreviate given names, so a raw string
    union leaves every author listed twice. Key on surname plus first initial,
    which collapses the abbreviations without merging genuine namesakes.
    """
    parts = [p for p in normalise(name).split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]}|{parts[0][0]}"


def _merge_lists(claims: list[Claim]) -> tuple[list, float]:
    """Union list fields, ordered by how many sources vouch for each item."""
    people = claims[0].field_name in {"authors", "affiliations"} if claims else False
    keyfn = person_key if claims and claims[0].field_name == "authors" else normalise

    votes: dict[str, float] = defaultdict(float)
    display: dict[str, Any] = {}
    for c in claims:
        for item in c.value:
            key = keyfn(item)
            if not key:
                continue
            votes[key] += c.confidence
            # keep the fullest spelling we have seen for this person
            current = display.get(key)
            if current is None or (people and len(str(item)) > len(str(current))):
                display[key] = item
    if not votes:
        return [], 0.0

    if people:
        # Author order is meaningful — first author is not just the most
        # corroborated one. Take the most trusted source's ordering as the
        # spine and append anyone it missed.
        spine = max(claims, key=lambda c: (c.confidence, len(c.value)))
        ordered = []
        seen = set()
        for item in spine.value:
            k = keyfn(item)
            if k and k not in seen:
                ordered.append(k)
                seen.add(k)
        ordered += [k for k in votes if k not in seen]
    else:
        ordered = sorted(votes, key=lambda k: -votes[k])

    top = max(votes.values())
    # agreement = how evenly the list is corroborated across sources
    agreement = sum(min(1.0, votes[k] / top) for k in ordered) / len(ordered)
    return [display[k] for k in ordered], round(agreement, 3)


def _pick_scalar(claims: list[Claim]) -> tuple[Any, Claim, float, bool]:
    groups: dict[Any, list[Claim]] = defaultdict(list)
    for c in claims:
        groups[normalise(c.value)].append(c)

    def group_score(key: Any) -> float:
        g = groups[key]
        # corroboration matters, but with diminishing returns — two mediocre
        # sources agreeing should not beat one authoritative source outright.
        best = max(x.confidence for x in g)
        return best + 0.08 * (len(g) - 1)

    winner_key = max(groups, key=group_score)
    winner = max(groups[winner_key], key=lambda x: x.confidence)
    agreement = len(groups[winner_key]) / len(claims)
    contested = len(groups) > 1
    return winner.value, winner, round(agreement, 3), contested


def resolve_record(rec: PaperRecord) -> PaperRecord:
    by_field: dict[str, list[Claim]] = defaultdict(list)
    for c in rec.claims:
        by_field[c.field_name].append(c)

    for name, claims in by_field.items():
        if name in LIST_FIELDS:
            value, agreement = _merge_lists(claims)
            if not value:
                continue
            best = max(claims, key=lambda x: x.confidence)
            conf = best.confidence * (0.85 + 0.15 * agreement)
            rec.resolved[name] = Resolution(
                field_name=name,
                value=value,
                source="+".join(sorted({c.source for c in claims})),
                confidence=round(min(0.99, conf), 3),
                agreement=agreement,
                contested=agreement < 0.6,
                alternatives=claims,
            )
            continue

        value, winner, agreement, contested = _pick_scalar(claims)
        if name in NUMERIC_FIELDS and value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        # corroboration bonus, capped
        conf = min(0.99, winner.confidence + 0.10 * (agreement * len(claims) - 1))
        if contested:
            conf *= 0.88
        rec.resolved[name] = Resolution(
            field_name=name,
            value=value,
            source=winner.source,
            confidence=round(max(0.05, conf), 3),
            agreement=agreement,
            contested=contested,
            # only the claims that actually lost on value, not the ones that
            # agreed and merely came from a less trusted source
            alternatives=[c for c in claims if normalise(c.value) != normalise(value)],
        )

    _sanity_checks(rec)
    return rec


def _sanity_checks(rec: PaperRecord) -> None:
    """Cheap rules that catch the failure modes structured sources actually have."""
    year = rec.get("year")
    if isinstance(year, int) and not (1600 <= year <= 2100):
        rec.errors.append(f"implausible year: {year}")
        rec.resolved.pop("year", None)

    if (cites := rec.get("citation_count")) is not None and isinstance(cites, int) and cites < 0:
        rec.errors.append("negative citation count")

    title = rec.get("title")
    if title and len(title) < 6:
        rec.errors.append("suspiciously short title")

    if rec.get("open_access") and not rec.get("oa_url"):
        rec.errors.append("marked open access but no OA location found")

    # A title-matched row with no DOI is the classic silent mismatch.
    if rec.ref_type == "title" and not rec.get("doi"):
        rec.errors.append("matched by title only — verify this is the right paper")
