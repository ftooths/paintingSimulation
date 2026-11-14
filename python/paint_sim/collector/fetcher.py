"""Polite HTTP fetcher with robots.txt, per-host rate limiting, retries, SHA-256 disk cache, and size limits."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

# Official httpx docs: https://github.com/encode/httpx/blob/master/docs/advanced/transports.md
import httpx

from paint_sim.collector.sources import load_collector_settings, normalize_url

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FETCH_CACHE_DIR = _REPO_ROOT / "sample" / "collector_jobs" / "_http_cache"


class FetchError(RuntimeError):
    """Raised when a URL fails politeness, content-type, payload-size, or HTTP status checks."""


class RobotsDisallowedError(FetchError):
    """Raised when `robots.txt` disallows fetching the requested URL."""


class PayloadTooLargeError(FetchError):
    """Raised when a response exceeds `max_payload_bytes`."""


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    content: bytes
    sha256: str
    from_cache: bool


class PoliteFetcher:
    """
    HTTP fetcher built on `httpx.Client` (Section 1.1, 2.4):
      - Checks `robots.txt` per host via `urllib.robotparser.RobotFileParser`.
      - Enforces per-host rate limits (`rate_limit_requests_per_sec`, default 1.0 req/s).
      - Caches responses on disk keyed by `SHA256(normalized_url)`.
      - Rejects responses exceeding `max_payload_bytes` (via `Content-Length` and stream chunk counting).
      - Validates `Content-Type` (`text/html`, `application/xhtml+xml`, `image/*`).
      - Supports injectable `transport` (`httpx.MockTransport`) and `sleep_fn`/`clock_fn` for deterministic tests.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit_per_sec: Optional[float] = None,
        max_payload_bytes: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        max_retries: Optional[int] = None,
        respect_robots_txt: Optional[bool] = None,
        user_agent: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = load_collector_settings()
        self.cache_dir = Path(cache_dir or DEFAULT_FETCH_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_sec = float(
            rate_limit_per_sec if rate_limit_per_sec is not None else cfg.get("rate_limit_requests_per_sec", 1.0)
        )
        self.max_payload_bytes = int(
            max_payload_bytes if max_payload_bytes is not None else cfg.get("max_payload_bytes", 26214400)
        )
        self.timeout_sec = float(
            timeout_sec if timeout_sec is not None else cfg.get("request_timeout_sec", 30.0)
        )
        self.max_retries = int(
            max_retries if max_retries is not None else cfg.get("max_retries", 2)
        )
        self.respect_robots_txt = bool(
            respect_robots_txt if respect_robots_txt is not None else cfg.get("respect_robots_txt", True)
        )
        self.user_agent = str(
            user_agent or cfg.get("user_agent", "PaintSimSampleCollector/1.0 (+http://127.0.0.1:8765; local-research)")
        )
        self._transport = transport
        self._clock = clock_fn
        self._sleep = sleep_fn
        self._last_request_by_host: Dict[str, float] = {}
        self._robots_by_host: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._client: Optional[httpx.Client] = None
        self.network_request_count: int = 0

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "PoliteFetcher":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def url_cache_key(url: str) -> str:
        norm = normalize_url(url)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def _get_cache_paths(self, url_key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{url_key}.bin", self.cache_dir / f"{url_key}.meta.json"

    def _read_cache(self, url: str) -> Optional[FetchResult]:
        key = self.url_cache_key(url)
        bin_path, meta_path = self._get_cache_paths(key)
        if not (bin_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data = bin_path.read_bytes()
            if len(data) > self.max_payload_bytes:
                raise PayloadTooLargeError(
                    f"Cached payload for {url} ({len(data)} bytes) exceeds max_payload_bytes ({self.max_payload_bytes})."
                )
            return FetchResult(
                url=url,
                final_url=meta.get("final_url", url),
                status_code=int(meta.get("status_code", 200)),
                content_type=str(meta.get("content_type", "")),
                content=data,
                sha256=meta.get("sha256") or hashlib.sha256(data).hexdigest(),
                from_cache=True,
            )
        except PayloadTooLargeError:
            raise
        except Exception as exc:
            logger.warning("Corrupt HTTP cache entry for %s (%s); refetching.", url, exc)
            return None

    def _write_cache(self, url: str, result: FetchResult) -> None:
        key = self.url_cache_key(url)
        bin_path, meta_path = self._get_cache_paths(key)
        bin_path.write_bytes(result.content)
        meta_path.write_text(
            json.dumps(
                {
                    "url": result.url,
                    "final_url": result.final_url,
                    "status_code": result.status_code,
                    "content_type": result.content_type,
                    "sha256": result.sha256,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _wait_rate_limit(self, host: str) -> None:
        if self.rate_limit_per_sec <= 0:
            return
        min_interval = 1.0 / self.rate_limit_per_sec
        now = self._clock()
        last = self._last_request_by_host.get(host)
        if last is not None:
            elapsed = now - last
            if elapsed < min_interval:
                self._sleep(min_interval - elapsed)
        self._last_request_by_host[host] = self._clock()

    def _get_client(self) -> httpx.Client:
        # Reuse a persistent httpx.Client for connection pooling across multiple fetches.
        # Source: https://github.com/encode/httpx/blob/master/docs/advanced/transports.md
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                transport=self._transport,
                headers={"User-Agent": self.user_agent},
                timeout=httpx.Timeout(self.timeout_sec),
                follow_redirects=True,
            )
        return self._client

    def _check_robots_txt(self, client: httpx.Client, url: str) -> None:
        if not self.respect_robots_txt:
            return
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_by_host.get(host_key)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{host_key}/robots.txt"
            try:
                self._wait_rate_limit(parsed.netloc.lower())
                resp = client.get(robots_url)
                if resp.status_code == 200 and resp.text:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.allow_all = True
            except Exception as exc:
                logger.debug("robots.txt unavailable for %s (%s); defaulting to allow.", host_key, exc)
                rp.allow_all = True
            self._robots_by_host[host_key] = rp

        if not rp.can_fetch(self.user_agent, url) and not rp.can_fetch("*", url):
            raise RobotsDisallowedError(f"Blocked by robots.txt: {url}")

    def fetch(
        self,
        url: str,
        *,
        allowed_content_types: tuple[str, ...] = ("text/html", "application/xhtml+xml", "image/"),
        use_cache: bool = True,
    ) -> FetchResult:
        """
        Fetch a URL politely, returning `FetchResult`.
        Raises `RobotsDisallowedError`, `PayloadTooLargeError`, or `FetchError` on failure.
        """
        norm = normalize_url(url)
        if use_cache:
            cached = self._read_cache(norm)
            if cached is not None:
                return cached

        parsed = urlparse(norm)
        host = parsed.netloc.lower()
        client = self._get_client()

        self._check_robots_txt(client, norm)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_rate_limit(host)
                self.network_request_count += 1
                with client.stream("GET", norm) as resp:
                    if resp.status_code >= 500 and attempt < self.max_retries:
                        last_exc = FetchError(f"HTTP {resp.status_code} for {norm}")
                        self._sleep(0.2 * (attempt + 1))
                        continue
                    if resp.status_code >= 400:
                        raise FetchError(f"HTTP {resp.status_code} fetching {norm}")

                    ctype_raw = resp.headers.get("content-type", "").lower()
                    ctype = ctype_raw.split(";")[0].strip()
                    if allowed_content_types and ctype:
                        if not any(ctype == pat or ctype.startswith(pat) for pat in allowed_content_types):
                            raise FetchError(
                                f"Unsupported Content-Type {ctype!r} for {norm} (expected {allowed_content_types})"
                            )

                    cl_header = resp.headers.get("content-length")
                    if cl_header and cl_header.isdigit() and int(cl_header) > self.max_payload_bytes:
                        raise PayloadTooLargeError(
                            f"Content-Length {cl_header} exceeds max_payload_bytes ({self.max_payload_bytes}) for {norm}"
                        )

                    chunks = []
                    total = 0
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > self.max_payload_bytes:
                            raise PayloadTooLargeError(
                                f"Streamed payload ({total} bytes) exceeds max_payload_bytes ({self.max_payload_bytes}) for {norm}"
                            )
                        chunks.append(chunk)

                    body = b"".join(chunks)
                    sha = hashlib.sha256(body).hexdigest()
                    res = FetchResult(
                        url=norm,
                        final_url=str(resp.url),
                        status_code=resp.status_code,
                        content_type=ctype or "application/octet-stream",
                        content=body,
                        sha256=sha,
                        from_cache=False,
                    )
                    if use_cache:
                        self._write_cache(norm, res)
                    return res
            except (RobotsDisallowedError, PayloadTooLargeError):
                raise
            except FetchError as exc:
                last_exc = exc
                if "HTTP 4" in str(exc) or "Unsupported Content-Type" in str(exc):
                    raise
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    self._sleep(0.2 * (attempt + 1))
                    continue
        raise FetchError(f"Failed to fetch {norm} after {self.max_retries + 1} attempts: {last_exc}") from last_exc
