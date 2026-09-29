"""
A stand-in for the OCRM lookup, so the whole scenario can be shown end to end.

This is NOT the bank's OCRM. It is a JSON file with invented vehicles, loaded
at start, and it exists because the demo has to show what happens after a plate
is confirmed: the card of the vehicle, or a plain statement that this plate is
not in the database. Every record in it is made up. No real borrower, credit or
vehicle is described here, and nothing is ever sent outside this process.

The distinction the scenario turns on is kept here as well: a plate that is not
found is a normal answer, not an error and not a recognition failure. So the
lookup answers 200 with found=false rather than 404, and the caller is expected
to say so in those words.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "test_ocrm.json"

_LOCK = threading.Lock()
_VEHICLES = {}
_SOURCE = None
_VISITS = []


def load(path=None):
    """Reads the test vehicles. Returns how many were loaded.

    A missing file is not fatal: the service still recognises plates, and the
    lookup then answers that no test database is configured, which is a clearer
    thing to see on screen than an empty card.
    """
    global _VEHICLES, _SOURCE
    path = Path(path) if path else DEFAULT_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        with _LOCK:
            _VEHICLES, _SOURCE = {}, None
        return 0

    vehicles = {}
    for item in raw.get("vehicles", []):
        plate = str(item.get("plate", "")).upper().strip()
        if plate:
            vehicles[plate] = item
    with _LOCK:
        _VEHICLES, _SOURCE = vehicles, str(path)
    return len(vehicles)


def configured():
    with _LOCK:
        return _SOURCE is not None


def lookup(plate):
    """{"found": bool, ...} for one plate. Never raises."""
    key = str(plate or "").upper().strip()
    with _LOCK:
        if _SOURCE is None:
            return {"found": False, "plate": key, "database": "not_configured"}
        record = _VEHICLES.get(key)
    if record is None:
        return {"found": False, "plate": key, "database": "test"}
    answer = dict(record)
    answer["found"] = True
    answer["database"] = "test"
    return answer


def create_visit(plate, note=""):
    """Records a field visit in memory and returns it. Test data, like the rest."""
    key = str(plate or "").upper().strip()
    with _LOCK:
        visit = {
            "visit_id": f"TEST-VISIT-{len(_VISITS) + 1:04d}",
            "plate": key,
            "note": str(note or "")[:500],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "database": "test",
        }
        _VISITS.append(visit)
    return visit


def visits():
    with _LOCK:
        return list(_VISITS)
