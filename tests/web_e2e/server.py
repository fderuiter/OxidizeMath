"""Local HTTP static server for Math Explorer WebAssembly E2E tests."""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


@dataclass
class RecordedRequest:
    """Captured HTTP request metadata."""

    method: str
    path: str
    status: int


class DistRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler serving math_explorer_gui/dist with optional COOP/COEP headers."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".ico": "image/x-icon",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".webmanifest": "application/manifest+json; charset=utf-8",
    }

    def __init__(
        self,
        *args,
        directory: str,
        isolated: bool,
        request_log: List[RecordedRequest],
        request_lock: threading.Lock,
        status_overrides: Dict[str, int],
        **kwargs,
    ) -> None:
        self._isolated = isolated
        self._request_log = request_log
        self._request_lock = request_lock
        self._status_overrides = status_overrides
        self._last_status = 200
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        clean_path = self.path.split("?", 1)[0]
        with self._request_lock:
            forced = self._status_overrides.get(clean_path)
        if forced is not None:
            body = f"HTTP {forced}".encode("utf-8")
            self.send_response(forced)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def send_response(self, code: int, message: str | None = None) -> None:
        self._last_status = code
        super().send_response(code, message)
        with self._request_lock:
            self._request_log.append(
                RecordedRequest(method=self.command or "GET", path=self.path, status=code)
            )

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        if self._isolated:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Silence per-request stderr noise during automated pytest execution.
        return


@dataclass
class LocalDistServer:
    """Running background ThreadingHTTPServer wrapper."""

    httpd: ThreadingHTTPServer
    thread: threading.Thread
    isolated: bool
    requests: List[RecordedRequest] = field(default_factory=list)
    status_overrides: Dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def host(self) -> str:
        return str(self.httpd.server_address[0])

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def snapshot_requests(self) -> List[Tuple[str, str, int]]:
        with self.lock:
            return [(r.method, r.path, r.status) for r in self.requests]

    def clear_requests(self) -> None:
        with self.lock:
            self.requests.clear()

    @contextmanager
    def override_status(self, path: str, status: int) -> Iterator[None]:
        normalized = path if path.startswith("/") else f"/{path}"
        with self.lock:
            self.status_overrides[normalized] = status
        try:
            yield
        finally:
            with self.lock:
                self.status_overrides.pop(normalized, None)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5.0)


def start_dist_server(dist_dir: Path, *, isolated: bool) -> LocalDistServer:
    """Start a background HTTP server on an ephemeral 127.0.0.1 port."""
    if not dist_dir.is_dir():
        raise FileNotFoundError(
            f"Distribution directory '{dist_dir}' does not exist. "
            "Build it first via 'cargo run -p xtask -- web-e2e' or 'trunk build --release'."
        )

    requests: List[RecordedRequest] = []
    status_overrides: Dict[str, int] = {}
    lock = threading.Lock()
    handler_factory = functools.partial(
        DistRequestHandler,
        directory=str(dist_dir.resolve()),
        isolated=isolated,
        request_log=requests,
        request_lock=lock,
        status_overrides=status_overrides,
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(
        target=httpd.serve_forever,
        name=f"dist-http-{'isolated' if isolated else 'static'}",
        daemon=True,
    )
    thread.start()
    return LocalDistServer(
        httpd=httpd,
        thread=thread,
        isolated=isolated,
        requests=requests,
        status_overrides=status_overrides,
        lock=lock,
    )
