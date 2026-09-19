"""Google Sheets import/export via a service account.

Import: pull a column of references from a sheet researchers already maintain.
Export: push the research dataset and the review queue back as separate tabs, so
verification happens where the team already works.
"""
from __future__ import annotations

import json

import pandas as pd

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _client(service_account_json: str):
    import gspread
    from google.oauth2.service_account import Credentials

    info = json.loads(service_account_json) if isinstance(service_account_json, str) else service_account_json
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def read_references(service_account_json: str, sheet_url: str,
                    worksheet: str | None = None, column: str | None = None) -> list[str]:
    gc = _client(service_account_json)
    sh = gc.open_by_url(sheet_url)
    ws = sh.worksheet(worksheet) if worksheet else sh.sheet1
    rows = ws.get_all_records()
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if column and column in df.columns:
        target = column
    else:
        # pick whichever column looks most like an identifier
        prefer = ["doi", "DOI", "url", "arxiv", "reference", "title", "Title"]
        target = next((c for c in prefer if c in df.columns), df.columns[0])
    return [str(v).strip() for v in df[target].dropna() if str(v).strip()]


def write_dataset(service_account_json: str, sheet_url: str,
                  frames: dict[str, pd.DataFrame]) -> str:
    """Write one worksheet per frame. Existing tabs with the same name are replaced."""
    import gspread

    gc = _client(service_account_json)
    sh = gc.open_by_url(sheet_url)
    for name, df in frames.items():
        df = df.fillna("").astype(str)
        try:
            ws = sh.worksheet(name)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=name, rows=max(len(df) + 10, 100),
                                  cols=max(len(df.columns) + 2, 10))
        ws.update([df.columns.tolist()] + df.values.tolist())
    return sh.url
