"""Fetch the Indre-et-Loire boundary once; the result is committed.

Run: uv run --group pilot python -m pilot.fetch_boundary
"""

import hashlib
import json

import requests

from pilot.common import BOUNDARY

API_URL = "https://geo.api.gouv.fr/departements/37?format=geojson&geometry=contour"
# Used only when the API answers without a geometry (it did on 2026-10-30).
FALLBACK_URL = (
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/"
    "departements/37-indre-et-loire/departement-37-indre-et-loire.geojson"
)


def fetch() -> tuple[bytes, str]:
    r = requests.get(API_URL, timeout=60)
    r.raise_for_status()
    if "geometry" in r.json():
        return r.content, API_URL
    r = requests.get(FALLBACK_URL, timeout=60)
    r.raise_for_status()
    return r.content, FALLBACK_URL


if __name__ == "__main__":
    content, url = fetch()
    geometry = json.loads(content)["geometry"]
    assert geometry["type"] in ("Polygon", "MultiPolygon")
    BOUNDARY.parent.mkdir(parents=True, exist_ok=True)
    BOUNDARY.write_bytes(content)
    print(url)
    print(hashlib.sha256(content).hexdigest())
