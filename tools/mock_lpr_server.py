#!/usr/bin/env python3
"""
Mock LPR server for integration work.

Speaks exactly the same HTTP contract as the real service, but recognizes
nothing: it replays a fixed sequence of responses. That lets the OCRM side be
written and tested before the real service is reachable over the network.

No GPU, no dependencies, no project files: one standard-library Python file.

    python3 mock_lpr_server.py               # listens on 0.0.0.0:8765
    python3 mock_lpr_server.py --port 9000

What it does:

    POST /frame?session_id=<id>    body: a JPEG frame (raw bytes)

    Every request for a session moves one step along the scenario below, so
    sending the same frame over and over walks through: nothing recognized,
    a plate confirmed, the same plate held, the plate cleared, a second
    (two-row) plate, and so on. The scenario repeats.

    Add &step=<n> to get one specific step instead, which is what you want in
    Postman: the same request always returns the same thing.

    GET /                          health check
    GET /scenario                  the whole scenario as JSON, for reference

The plates here are made up. They have the right shape for Kazakhstan
(3 digits, 3 letters, 2 digits) but are not real plates, so nothing can
accidentally match a real customer.

Validation is real, so error handling can be tested:
    no Content-Length, bad session_id           -> 400 bad_request
    body that is not a JPEG                     -> 400 decode_failed
    body over 10 MB                             -> 413 payload_too_large
    any path other than /frame                  -> 404 not_found
    session_id "unavailable"                    -> 503 worker_unavailable
"""

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_FRAME_BYTES = 10 * 1024 * 1024
SESSION_ID_RE = re.compile(r"[A-Za-z0-9._:@-]{1,128}")
PORT = 8765


def response(plate="", confirmed=False, changed=False, bbox=None, confidence=0.0,
             ocr_confidence=None, plate_type=None, raw_text=None, comment=""):
    return {
        "ok": True,
        "plate": plate,
        "confirmed": confirmed,
        "changed": changed,
        "bbox": bbox,
        "confidence": confidence,
        "ocr_confidence": ocr_confidence,
        "plate_type": plate_type,
        "raw_text": raw_text,
        "_comment": comment,
    }


# One pass of a realistic session: the employee walks up to a car, the plate is
# confirmed, held while the camera stays on it, cleared when the camera moves
# away, then the next car, which has a two-row plate.
SCENARIO = [
    response(comment="no plate in the frame yet"),
    response(bbox=[610, 880, 980, 985, 0.71], confidence=0.71, ocr_confidence=0.44,
             plate_type="normal", raw_text="01ABC",
             comment="a plate is visible but not read well enough to confirm"),
    response(plate="123ABC01", confirmed=True, changed=True,
             bbox=[604, 876, 986, 988, 0.93], confidence=0.93, ocr_confidence=0.96,
             plate_type="normal", raw_text="123ABC01",
             comment="CONFIRMED: changed=true, this is when OCRM should search the database"),
    response(plate="123ABC01", confirmed=True,
             bbox=[607, 878, 984, 986, 0.94], confidence=0.94, ocr_confidence=0.97,
             plate_type="normal", raw_text="123ABC01",
             comment="same car still in view: changed=false, do not search again"),
    response(plate="123ABC01", confirmed=True,
             bbox=[612, 881, 979, 984, 0.90], confidence=0.90, ocr_confidence=0.92,
             plate_type="normal", raw_text="123ABC01",
             comment="still the same car"),
    response(plate="123ABC01", confirmed=True, comment="camera moved away; the plate is still held"),
    response(comment="CLEARED: not read for 2 s, so plate is empty and confirmed=false. "
                     "OCRM should hide the vehicle card. changed stays false"),
    response(bbox=[700, 900, 900, 1060, 0.66], confidence=0.66, ocr_confidence=0.51,
             plate_type="square", raw_text="456 / 02DEF",
             comment="next car, a two-row plate, being read"),
    response(plate="456DEF02", confirmed=True, changed=True,
             bbox=[698, 898, 903, 1063, 0.91], confidence=0.91, ocr_confidence=0.94,
             plate_type="square", raw_text="456 / 02DEF",
             comment="CONFIRMED a two-row plate: rows read as top / bottom"),
    response(plate="456DEF02", confirmed=True,
             bbox=[699, 899, 902, 1061, 0.92], confidence=0.92, ocr_confidence=0.95,
             plate_type="square", raw_text="456 / 02DEF",
             comment="same car"),
]

_steps = {}
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def _send(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._send({"ok": False, "error": message, "error_code": code}, status)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/scenario":
            self._send({"steps": len(SCENARIO), "scenario": SCENARIO})
        elif path == "/":
            with _lock:
                count = len(_steps)
            self._send({"ok": True, "service": "lpr-api-server (mock)", "port": PORT,
                        "session_count": count, "ocr_variant_mode": "mock",
                        "worker_restarts": {"yolo": 0, "ocr": 0}})
        else:
            self._error(404, "not_found", "Use POST /frame")

    def do_POST(self):
        path, _, query_string = self.path.partition("?")
        if path != "/frame":
            self._error(404, "not_found", "Use POST /frame")
            return

        query = {}
        for part in query_string.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                query[k] = v
        session_id = query.get("session_id", "default")

        if not SESSION_ID_RE.fullmatch(session_id):
            self._error(400, "bad_request",
                        "session_id must be 1-128 characters: letters, digits, . _ : @ -")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(400, "bad_request", "Content-Length header is required")
            return
        try:
            length = int(raw_length)
        except ValueError:
            self._error(400, "bad_request", "invalid Content-Length")
            return
        if length <= 0:
            self._error(400, "bad_request", "empty request body")
            return
        if length > MAX_FRAME_BYTES:
            self._error(413, "payload_too_large",
                        f"frame is {length} bytes, limit is {MAX_FRAME_BYTES}")
            return

        data = self.rfile.read(length)
        if not data.startswith(b"\xff\xd8\xff"):
            self._error(400, "decode_failed", "body is not a decodable image")
            return

        if session_id == "unavailable":
            self._error(503, "worker_unavailable", "OCR worker timed out")
            return

        if "step" in query:
            try:
                index = int(query["step"]) % len(SCENARIO)
            except ValueError:
                self._error(400, "bad_request", "step must be a number")
                return
        else:
            with _lock:
                index = _steps.get(session_id, 0)
                _steps[session_id] = (index + 1) % len(SCENARIO)

        result = dict(SCENARIO[index])
        result["session_id"] = session_id
        result["processing_time_ms"] = 38.0
        result["frame_size"] = [1280, 720]
        result["ocr_variant_mode"] = "mock"
        result["_step"] = index
        self._send(result)

    def log_message(self, format, *args):
        print(f"[mock] {self.address_string()} {format % args}", flush=True)


def main():
    global PORT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    PORT = args.port

    print("=" * 62)
    print("LPR MOCK SERVER — replays a fixed sequence, recognizes nothing")
    print("=" * 62)
    print(f"POST http://{args.host}:{PORT}/frame?session_id=<id>   body: a JPEG frame")
    print(f"GET  http://{args.host}:{PORT}/                        health check")
    print(f"GET  http://{args.host}:{PORT}/scenario                the whole sequence")
    print(f"\n{len(SCENARIO)} steps; add &step=<n> to ask for one of them directly.")
    print("=" * 62, flush=True)

    server = ThreadingHTTPServer((args.host, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
