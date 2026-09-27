from __future__ import annotations

import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path


class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def serve(
    repository_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8788,
    open_browser: bool = True,
) -> None:
    root = Path(repository_root).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    url = f"http://{host}:{port}/web/"
    with ReusableThreadingTCPServer((host, port), handler) as httpd:
        print(f"US Tech Monitor: {url}")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
