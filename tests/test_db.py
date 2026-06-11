"""Tests for the Supabase connection."""

import os

import pytest

from stock_picker import db


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SUPABASE_URL and SUPABASE_KEY"):
        db._credentials()


@pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")),
    reason="Supabase credentials not configured (.env)",
)
def test_connection():
    db.check_connection()