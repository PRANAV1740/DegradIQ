"""Robust, resumable rate limiter and checkpoint manager for FastF1 ingestion.

Enforces:
- Minimum interval between uncached outbound HTTP requests (default 9.0s).
- Zero artificial delay on cache hits.
- Origin 'Retry-After' header compliance.
- Exponential backoff with random jitter on HTTP 429, 503, and RateLimitExceededError (up to 5 retries).
- Atomic session checkpoints under degradiq/cache/hazard_checkpoints/session_{year}_{round}.parquet.
"""
from __future__ import annotations

import email.utils
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

import fastf1
from fastf1 import req
import pandas as pd

logger = logging.getLogger("degradiq.rate_limiter")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [RateLimiter] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SCHEMA_VERSION = "1.0"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "cache" / "hazard_checkpoints"


def parse_retry_after(header_val: str | None) -> float | None:
    """Parse HTTP Retry-After header into seconds to wait."""
    if not header_val:
        return None
    val = header_val.strip()
    if val.isdigit():
        return max(0.0, float(val))
    try:
        dt = email.utils.parsedate_to_datetime(val)
        diff = (dt.timestamp() - time.time())
        return max(0.0, float(diff))
    except Exception:
        return None


class RateLimiter:
    """Outbound HTTP rate limiter and retry handler."""

    def __init__(
        self,
        min_interval_s: float = 9.0,
        max_retries: int = 5,
        base_backoff_s: float = 4.0,
        jitter_s: float = 2.0,
    ):
        self.min_interval_s = float(min_interval_s)
        self.max_retries = max_retries
        self.base_backoff_s = float(base_backoff_s)
        self.jitter_s = float(jitter_s)
        self.last_outbound_time: float = 0.0
        self.total_outbound_requests: int = 0
        self.total_cache_hits: int = 0
        self.total_retries: int = 0

    def pace_outbound_request(self) -> float:
        """Pace uncached outbound HTTP requests by waiting if needed."""
        now = time.time()
        elapsed = now - self.last_outbound_time
        slept = 0.0
        if self.last_outbound_time > 0 and elapsed < self.min_interval_s:
            slept = self.min_interval_s - elapsed
            logger.info("Pacing outbound request: waiting %.2fs (min_interval=%.1fs)", slept, self.min_interval_s)
            time.sleep(slept)
        self.last_outbound_time = time.time()
        self.total_outbound_requests += 1
        return slept

    def compute_backoff(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Calculate exponential backoff delay with random jitter."""
        if retry_after_s is not None and retry_after_s > 0:
            # Respect server Retry-After plus a small random jitter
            delay = retry_after_s + random.uniform(0.5, 1.5)
            logger.info("Server requested Retry-After=%.2fs (total wait %.2fs)", retry_after_s, delay)
            return delay
        exp_delay = self.base_backoff_s * (2 ** (attempt - 1))
        jitter = random.uniform(0.0, self.jitter_s)
        delay = exp_delay + jitter
        logger.info("Calculated exponential backoff (attempt %d/%d): %.2fs", attempt, self.max_retries, delay)
        return delay

    def execute_with_retry(self, send_fn: Callable[[], Any], request_desc: str = "") -> Any:
        """Execute an outbound network call with rate pacing and exponential retry."""
        attempt = 0
        while True:
            self.pace_outbound_request()
            try:
                response = send_fn()
                # Check HTTP status codes for rate-limiting or temporary unavailability
                status_code = getattr(response, "status_code", 200)
                if status_code in (429, 503):
                    attempt += 1
                    if attempt > self.max_retries:
                        logger.error("Max retries (%d) exceeded for HTTP %d: %s", self.max_retries, status_code, request_desc)
                        return response
                    self.total_retries += 1
                    retry_header = None
                    if hasattr(response, "headers"):
                        retry_header = response.headers.get("Retry-After")
                    delay = self.compute_backoff(attempt, parse_retry_after(retry_header))
                    logger.warning(
                        "Received HTTP %d for %s. Retry %d/%d after %.2fs",
                        status_code, request_desc, attempt, self.max_retries, delay
                    )
                    time.sleep(delay)
                    continue
                return response
            except (req.RateLimitExceededError, Exception) as exc:
                is_rate_limit = isinstance(exc, req.RateLimitExceededError)
                is_transient = "503" in str(exc) or "429" in str(exc) or "reset" in str(exc).lower()
                if not (is_rate_limit or is_transient):
                    # Permanent data / parsing error: do not retry indefinitely
                    logger.error("Permanent or unrecoverable error for %s: %s: %s", request_desc, type(exc).__name__, exc)
                    raise
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Max retries (%d) exceeded on error for %s: %s", self.max_retries, request_desc, exc)
                    raise
                self.total_retries += 1
                delay = self.compute_backoff(attempt)
                logger.warning(
                    "Caught %s for %s. Retry %d/%d after %.2fs. Error: %s",
                    type(exc).__name__, request_desc, attempt, self.max_retries, delay, exc
                )
                time.sleep(delay)


def install_rate_limiter(limiter: RateLimiter | None = None) -> RateLimiter:
    """Install outbound request pacing into FastF1's session infrastructure.

    Only intercepts actual outbound network calls; cached responses bypass send
    and incur 0s pacing delay.
    """
    if limiter is None:
        limiter = RateLimiter()

    orig_send = req._SessionWithRateLimiting.send

    def instrumented_send(self, request, **kwargs):
        # Cache hits in requests_cache do not call super().send(); this is reached
        # only when an actual outbound HTTP request is about to leave the machine.
        url_snippet = str(getattr(request, "url", "unknown"))[:80]
        return limiter.execute_with_retry(
            lambda: orig_send(self, request, **kwargs),
            request_desc=url_snippet
        )

    req._SessionWithRateLimiting.send = instrumented_send
    return limiter


class CheckpointManager:
    """Durable, atomic session hazard-row checkpoints."""

    def __init__(self, checkpoint_dir: Path | str = CHECKPOINT_DIR):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, year: int, round_id: Any) -> Path:
        sanitized_round = str(round_id).replace(" ", "_").replace("/", "_")
        return self.checkpoint_dir / f"session_{year}_{sanitized_round}.parquet"

    def metadata_path(self, year: int, round_id: Any) -> Path:
        sanitized_round = str(round_id).replace(" ", "_").replace("/", "_")
        return self.checkpoint_dir / f"session_{year}_{sanitized_round}_meta.json"

    def has_valid_checkpoint(
        self,
        year: int,
        round_id: Any,
        expected_columns: list[str],
        event_name: str | None = None,
    ) -> bool:
        """Validate checkpoint before reuse, checking round_id then event_name."""
        for candidate_id in [round_id, event_name]:
            if candidate_id is None:
                continue
            mpath = self.metadata_path(year, candidate_id)
            if not mpath.exists():
                continue
            try:
                with open(mpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("schema_version") != SCHEMA_VERSION:
                    continue
                if meta.get("year") != year:
                    continue
                if meta.get("status") in ("no eligible rows", "unavailable"):
                    return True
                cpath = self.checkpoint_path(year, candidate_id)
                if not cpath.exists():
                    continue
                df = pd.read_parquet(cpath)
                has_all_cols = all(col in df.columns for col in expected_columns)
                if has_all_cols:
                    return True
            except Exception as exc:
                logger.warning("Corrupt or invalid checkpoint %s: %s", mpath.name, exc)
                continue
        return False

    def load_checkpoint(
        self,
        year: int,
        round_id: Any,
        event_name: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        for candidate_id in [round_id, event_name]:
            if candidate_id is None:
                continue
            cpath = self.checkpoint_path(year, candidate_id)
            mpath = self.metadata_path(year, candidate_id)
            if mpath.exists():
                with open(mpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("status") in ("no eligible rows", "unavailable") or not cpath.exists():
                    return pd.DataFrame(), meta
                df = pd.read_parquet(cpath)
                return df, meta
        return pd.DataFrame(), {}

    def save_checkpoint(
        self,
        year: int,
        round_id: Any,
        event_name: str,
        rows: pd.DataFrame,
        status: str = "loaded",
        error: str = "",
    ) -> None:
        """Atomically write parquet data and metadata sidecar."""
        cpath = self.checkpoint_path(year, round_id)
        mpath = self.metadata_path(year, round_id)
        tmp_cpath = cpath.with_suffix(".parquet.tmp")
        tmp_mpath = mpath.with_suffix(".json.tmp")

        meta = {
            "schema_version": SCHEMA_VERSION,
            "year": year,
            "round": str(round_id),
            "event_name": event_name,
            "status": status,
            "row_count": len(rows) if rows is not None else 0,
            "positive_count": int(rows["StopNextLap"].sum()) if rows is not None and "StopNextLap" in rows.columns else 0,
            "error": error,
            "saved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }

        # Write parquet atomically
        if rows is not None and not rows.empty:
            rows.to_parquet(tmp_cpath, index=False)
            os.replace(tmp_cpath, cpath)
        elif cpath.exists():
            # If empty and file existed, remove
            try:
                cpath.unlink()
            except OSError:
                pass

        # Write JSON metadata atomically
        with open(tmp_mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_mpath, mpath)
        logger.info("Saved atomic checkpoint for %d Round %s (%s): %d rows, status=%s", year, round_id, event_name, meta["row_count"], status)
