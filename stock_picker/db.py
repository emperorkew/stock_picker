"""Supabase connection setup.

Reads credentials from a .env file (or the environment) and exposes a
single shared client via get_client().

Run this module directly to test the connection:

    python -m stock_picker.db
"""

import os
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def _credentials() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set. "
            "Copy .env.example to .env and fill in your project credentials."
        )
    return url, key


@lru_cache(maxsize=1)
def get_client() -> Client:
    url, key = _credentials()
    return create_client(url, key)


def check_connection(timeout: float = 10.0) -> None:
    """Verify the Supabase URL is reachable and the API key is accepted.

    Hits the PostgREST root endpoint, which answers 200 for any valid key
    without needing a specific table to exist. Raises on failure.
    """
    url, key = _credentials()
    response = httpx.get(
        f"{url.rstrip('/')}/rest/v1/",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    response.raise_for_status()


if __name__ == "__main__":
    try:
        check_connection()
    except Exception as exc:  # noqa: BLE001 - report any failure to the user
        raise SystemExit(f"Supabase connection FAILED: {exc}")
    print("Supabase connection OK")
