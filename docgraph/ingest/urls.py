from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse


def parse_url_lines(text: str) -> list[str]:
    """Parse newline-separated URLs; skip blanks and # comments; dedupe."""
    urls: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            urls.append(line)
    return urls


def url_display_name(url: str) -> str:
    """Derive a filesystem-safe display name from a URL."""
    parsed = urlparse(url)
    host = parsed.hostname or "page"
    path = parsed.path.strip("/").replace("/", "_") or "index"
    name = f"{host}_{path}"
    name = re.sub(r"[^\w.\-]", "_", name)
    return name[:120] or "page"


def validate_url(url: str) -> str:
    """Validate http(s) URL and block SSRF targets (localhost, private IPs)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme (http/https only): {url}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"invalid URL: {url}")

    lowered = host.lower()
    if lowered in ("localhost", "0.0.0.0") or lowered.endswith(".local"):
        raise ValueError(f"blocked host: {url}")

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"blocked host: {url}")
    except ValueError as exc:
        if "blocked host" in str(exc):
            raise
        # hostname, not an IP literal — allowed

    return url
