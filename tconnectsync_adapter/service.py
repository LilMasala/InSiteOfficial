from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .worker import TandemIngestionService


_SERVICE: TandemIngestionService | None = None


def _service() -> TandemIngestionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = TandemIngestionService()
    return _SERVICE


class TandemAdapterHandler(BaseHTTPRequestHandler):
    server_version = "InSiteTandemAdapter/0.1"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def log_message(self, fmt: str, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "status": "ok"})
            return
        if parsed.path == "/telemetry/tandem/status":
            query = parse_qs(parsed.query)
            uid = (query.get("uid") or [None])[0]
            if not uid:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "`uid` is required"})
                return
            self._send_json(HTTPStatus.OK, _service().status(uid=uid))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/telemetry/tandem/connect":
                uid = payload["uid"]
                email = payload["email"]
                password = payload["password"]
                region = payload.get("region", "US")
                pump_serial_number = payload.get("pumpSerialNumber")
                result = _service().connect(
                    uid=uid,
                    email=email,
                    password=password,
                    region=region,
                    pump_serial_number=pump_serial_number,
                )
                self._send_json(HTTPStatus.OK, result)
                return
            if parsed.path == "/telemetry/tandem/sync":
                uid = payload["uid"]
                days = int(payload.get("days", 14))
                result = _service().sync(uid=uid, days=days)
                self._send_json(HTTPStatus.OK, result)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except KeyError as err:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"missing field: {err.args[0]}"})
        except Exception as err:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(err)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Tandem ingestion adapter service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TandemAdapterHandler)
    print(f"Tandem adapter listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
