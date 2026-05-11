# pyright: reportMissingModuleSource=false
"""Scrape OBS Bologna (JavaScript-driven) into core.Page via Selenium + headless Chrome."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Page
from core.obs_bologna_scraper import (
    SOURCE_LABEL,
    _is_prog_aspx_url,
    build_driver,
    collect_bologna_urls,
    fetch_page_extract,
    fetch_page_extract_http,
    run_backfill_obs_prog_tabs_http,
)

logger = logging.getLogger(__name__)


def _default_fetch_workers_from_env() -> int:
    """CLI default for --fetch-workers; override with OBS_FETCH_WORKERS (1–12)."""
    raw = (os.environ.get("OBS_FETCH_WORKERS") or "").strip()
    if raw.isdigit():
        return max(1, min(12, int(raw)))
    return 4


class Command(BaseCommand):
    help = (
        "Scrape obs.acibadem.edu.tr Bologna pages with Selenium (headless Chrome) "
        "and store them in core.Page. After a non-dry-run, runs HTTP backfill for "
        "synthesized prog*.aspx from DB showPac rows (same as backfill_obs_prog_tabs) "
        "unless --skip-prog-backfill."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--delay",
            type=float,
            default=1.0,
            help="Seconds to sleep after each navigation or JS-triggering click (default: 1.0).",
        )
        parser.add_argument(
            "--max-programs",
            type=int,
            default=None,
            help="Optional cap on total pages to fetch and save (including index).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not write to the database; print URL, title preview, and content length.",
        )
        parser.add_argument(
            "--skip-section",
            type=str,
            default="",
            help="Comma-separated substrings; skip expand clicks whose link text or href matches (case-insensitive).",
        )
        parser.add_argument(
            "--lang",
            type=str,
            default="en",
            help="Prefer this site language: drop index/section URLs with lang=tr when set to en (default: en).",
        )
        parser.add_argument(
            "--fetch-workers",
            type=int,
            default=_default_fetch_workers_from_env(),
            help=(
                "Parallel Chrome instances for URL fetch after discovery (default: 4, or "
                "OBS_FETCH_WORKERS). Use 6–8 in Docker if RAM allows. Max 12. Use 1 for "
                "sequential (low RAM). Each worker is one headless browser."
            ),
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help=(
                "Print discovery progress, Done summary, failures, and scraper WARNING logs. "
                "Default is quiet: only Saved lines (and [dry-run] lines when using --dry-run). "
                "(Short -v is reserved by Django for --verbosity.)"
            ),
        )
        parser.add_argument(
            "--progress",
            action="store_true",
            help=(
                "Print each URL before fetch and elapsed seconds after (Saved lines unchanged). "
                "Useful to see where a run is spending time without full --verbose."
            ),
        )
        parser.add_argument(
            "--stall-warn-interval",
            type=float,
            default=60.0,
            help=(
                "When using sequential Chrome (--fetch-workers 1), log a 'still waiting on this URL' "
                "line every N seconds while a single page load is in progress (0=off). "
                "Default: 60. Requires --progress or --verbose for visible output."
            ),
        )
        parser.add_argument(
            "--http-pass2-timeout",
            type=float,
            default=None,
            help=(
                "Max seconds per URL for second-pass parallel HTTP fetches (prog*.aspx). "
                "Overrides env OBS_HTTP_PASS2_PER_URL_TIMEOUT_S; omit to use env or default 180."
            ),
        )
        parser.add_argument(
            "--clear-existing",
            action="store_true",
            help=(
                "Delete existing OBS Bologna Page rows (source=obs.acibadem.edu.tr) and their "
                "chunks before scraping. ACU and other sources are unchanged."
            ),
        )
        parser.add_argument(
            "--fast",
            action="store_true",
            help=(
                "Aggressive speed preset: sets OBS_SHOWPAC_MAX_DYNCON, OBS_SHOWPAC_SIDEBAR_CLICKS, "
                "OBS_DYNCON_HTTP_TIMEOUT defaults (via setdefault) if you have not set them—fewer "
                "dynCon fetches and sidebar clicks; thinner content, shorter wall time. "
                "Combine with --delay 0.12–0.2 and high --fetch-workers (e.g. 8–12) when RAM allows. "
                "Network/server load varies; a fixed deadline (e.g. 10 minutes) is not guaranteed."
            ),
        )
        parser.add_argument(
            "--skip-prog-backfill",
            action="store_true",
            help=(
                "After a non-dry-run scrape, do not run the extra HTTP pass that synthesizes prog*.aspx "
                "URLs from showPac rows still in DB (same logic as manage.py backfill_obs_prog_tabs)."
            ),
        )

    def handle(self, *args, **options):
        style: Any = self.style
        page_objects: Any = getattr(Page, "objects")

        if bool(options.get("fast")):
            # User-supplied env wins; these only fill in missing OBS_* defaults.
            os.environ.setdefault("OBS_SHOWPAC_MAX_DYNCON", "6")
            os.environ.setdefault("OBS_SHOWPAC_SIDEBAR_CLICKS", "6")
            os.environ.setdefault("OBS_DYNCON_HTTP_TIMEOUT", "18")

        delay: float = max(0.0, float(options["delay"]))
        dry_run: bool = options["dry_run"]
        clear_existing: bool = bool(options.get("clear_existing"))
        max_programs: int | None = options["max_programs"]
        skip_raw: str = options["skip_section"] or ""
        skip_parts = [s.strip() for s in skip_raw.split(",") if s.strip()]
        fetch_workers: int = max(1, min(12, int(options["fetch_workers"] or 1)))
        target_lang: str = options.get("lang") or "en"
        verbose: bool = bool(options.get("verbose"))
        if verbose and bool(options.get("fast")):
            self.stdout.write(
                style.NOTICE(
                    "Fast preset: OBS_SHOWPAC_MAX_DYNCON="
                    f"{os.environ.get('OBS_SHOWPAC_MAX_DYNCON', '')} "
                    "OBS_SHOWPAC_SIDEBAR_CLICKS="
                    f"{os.environ.get('OBS_SHOWPAC_SIDEBAR_CLICKS', '')} "
                    "OBS_DYNCON_HTTP_TIMEOUT="
                    f"{os.environ.get('OBS_DYNCON_HTTP_TIMEOUT', '')} "
                    "(use setdefault—your env overrides preset)."
                )
            )
        progress_fetch: bool = verbose or bool(options.get("progress"))
        stall_interval: float = max(0.0, float(options.get("stall_warn_interval") or 0.0))

        _http_arg = options.get("http_pass2_timeout")
        if _http_arg is not None:
            http_pass2_timeout: float = max(5.0, float(_http_arg))
        else:
            _htt_env = (os.environ.get("OBS_HTTP_PASS2_PER_URL_TIMEOUT_S") or "").strip()
            try:
                http_pass2_timeout = max(5.0, float(_htt_env)) if _htt_env else 180.0
            except ValueError:
                http_pass2_timeout = 180.0

        _slow_env = (os.environ.get("OBS_SLOW_FETCH_WARN_SEC") or "").strip()
        try:
            slow_fetch_warn_sec: float = float(_slow_env) if _slow_env else 90.0
        except ValueError:
            slow_fetch_warn_sec = 90.0

        log_levels_restored: list[tuple[logging.Logger, int]] = []
        if not verbose:
            for name in (
                "core.obs_bologna_scraper",
                "urllib3",
                "urllib3.connectionpool",
            ):
                lg = logging.getLogger(name)
                log_levels_restored.append((lg, lg.level))
                lg.setLevel(logging.CRITICAL)

        if clear_existing:
            if dry_run:
                self.stdout.write(
                    style.WARNING(
                        "[dry-run] Skipping --clear-existing (no database changes)."
                    )
                )
            else:
                with transaction.atomic():
                    qs = Page.objects.filter(source=SOURCE_LABEL)
                    n = qs.count()
                    deleted, detail = qs.delete()
                self.stdout.write(
                    style.WARNING(
                        f"Cleared OBS pages: {n} row(s) removed "
                        f"(DocumentChunk cascade: {detail})."
                    )
                )

        drivers_to_quit: list[Any] = []
        saved = 0
        failed = 0
        urls: list[str] = []
        shutdown_event = threading.Event()
        monitor_thread: threading.Thread | None = None

        def _quit_all_drivers() -> None:
            for d in drivers_to_quit:
                try:
                    d.quit()
                except Exception as e:
                    if verbose:
                        logger.warning("driver.quit() failed: %s", e)
                    else:
                        logger.debug("driver.quit() failed: %s", e)

        def _out_flush() -> None:
            try:
                self.stdout.flush()
            except Exception:
                pass

        try:

            if verbose:
                self.stdout.write(
                    "Starting headless Chrome… "
                    "(set PYTHONUNBUFFERED=1 in Docker so logs appear immediately.)"
                )
                _out_flush()
            collect_driver = build_driver()
            drivers_to_quit.append(collect_driver)
            urls = collect_bologna_urls(
                collect_driver,
                delay,
                skip_section_parts=skip_parts,
                target_lang=target_lang,
            )
            if verbose:
                self.stdout.write(
                    style.NOTICE(
                        f"Collected {len(urls)} unique URLs. "
                        f"Fetching with {fetch_workers} worker(s) — "
                        "showPac pages are slowest (sidebar + dynCon HTTP)."
                    )
                )
                _out_flush()

            if max_programs is not None:
                cap = max(1, int(max_programs))
                urls = urls[:cap]
                if verbose:
                    self.stdout.write(style.WARNING(f"Capped to {cap} URLs (--max-programs)."))

            # showPac = Selenium sidebar + dynCon HTTP (slowest). Fetch lighter pages first so
            # Saved: lines appear sooner and parallel workers stay busier.
            urls.sort(key=lambda u: ("showpac" in (u or "").lower(), u or ""))

            def _save_one(url: str, title: str, text: str, embedding_units: list[str]) -> None:
                nonlocal saved
                if not text:
                    if verbose:
                        self.stdout.write(style.WARNING(f"Empty body text: {url}"))
                    else:
                        logger.debug("Empty body text: %s", url)
                if dry_run:
                    self.stdout.write(
                        f"[dry-run] {url} | {title[:80]!r} | {len(text)} chars"
                    )
                else:
                    page_objects.update_or_create(
                        url=url,
                        defaults={
                            "title": title,
                            "content": text,
                            "embedding_units": embedding_units or None,
                            "source": SOURCE_LABEL,
                        },
                    )
                    saved += 1
                    self.stdout.write(style.SUCCESS(f"Saved: {url}"))

            progress_lock = threading.Lock()
            stall_lock = threading.Lock()
            stall_state: dict[str, Any] = {"url": "", "start": 0.0, "last_k": 0}

            def _fmt_url(u: str, max_len: int = 120) -> str:
                s = u or ""
                if len(s) <= max_len:
                    return s
                return s[: max_len - 1] + "…"

            def _log_fetch_start(phase: str, idx: int, total: int, url: str) -> None:
                if not progress_fetch:
                    return
                msg = f"[{phase}] ({idx}/{total}) İstek → {_fmt_url(url)}"
                logger.info("%s", msg)
                with progress_lock:
                    self.stdout.write(style.NOTICE(msg))
                    _out_flush()

            def _log_fetch_done(
                url: str, elapsed: float, ok: bool, note: str = ""
            ) -> None:
                if elapsed >= slow_fetch_warn_sec:
                    logger.warning(
                        "slow OBS fetch %.1fs %s%s",
                        elapsed,
                        url,
                        f" ({note})" if note else "",
                    )
                if not progress_fetch:
                    return
                status = "tamam" if ok else "hata"
                line = f"  ← {status} {elapsed:.1f}s"
                if note:
                    line += f" · {note}"
                with progress_lock:
                    self.stdout.write(style.SUCCESS(line) if ok else style.WARNING(line))
                    _out_flush()

            def _begin_stall(u: str) -> None:
                with stall_lock:
                    stall_state["url"] = u
                    stall_state["start"] = time.monotonic()
                    stall_state["last_k"] = 0

            def _end_stall() -> None:
                with stall_lock:
                    stall_state["url"] = ""
                    stall_state["start"] = 0.0
                    stall_state["last_k"] = 0

            def _stall_monitor_loop() -> None:
                tick = (
                    min(10.0, max(1.0, stall_interval / 6.0))
                    if stall_interval
                    else 10.0
                )
                while not shutdown_event.wait(tick):
                    with stall_lock:
                        u = stall_state["url"]
                        t0 = stall_state["start"]
                        last_k = int(stall_state["last_k"])
                    if not u:
                        continue
                    elapsed = time.monotonic() - t0
                    if stall_interval <= 0 or elapsed < stall_interval:
                        continue
                    k = int(elapsed // stall_interval)
                    if k < 1:
                        continue
                    if k <= last_k:
                        continue
                    with stall_lock:
                        stall_state["last_k"] = k
                    msg = (
                        f"Bekleniyor (~{k * stall_interval:.0f}s bu URL'de): {_fmt_url(u)}"
                    )
                    logger.warning(msg)
                    with progress_lock:
                        self.stdout.write(style.WARNING(msg))
                        _out_flush()

            def _maybe_start_stall_monitor() -> None:
                nonlocal monitor_thread
                if monitor_thread is not None:
                    return
                if stall_interval <= 0 or not progress_fetch:
                    return
                monitor_thread = threading.Thread(
                    target=_stall_monitor_loop,
                    name="obs-stall-warn",
                    daemon=True,
                )
                monitor_thread.start()

            def _env_second_pass_cap() -> int:
                """OBS second-pass queue cap (sorted prefix). Raised default: each showPac adds ~17 synthetic prog URLs."""
                raw = (os.environ.get("OBS_SECOND_PASS_MAX") or "").strip()
                if raw.isdigit():
                    return max(10, min(20000, int(raw)))
                return 6000

            def _second_pass_notice(extra: set[str], first: set[str]) -> list[str]:
                cap = _env_second_pass_cap()
                cand = sorted(extra - first)[:cap]
                if verbose and cand:
                    self.stdout.write(
                        style.NOTICE(
                            f"Second pass: fetching {len(cand)} URL(s) found in page HTML "
                            f"(e.g. prog*.aspx)— not in initial discovery list."
                        )
                    )
                return cand

            def _use_http_second_pass(page_url: str) -> bool:
                """prog*.aspx pages are large static GETs—no Selenium needed (much faster)."""
                if "showpac" in (page_url or "").lower():
                    return False
                if _is_prog_aspx_url(page_url):
                    return True
                pl = (page_url or "").lower()
                return "prog" in pl and ".aspx" in pl

            def _run_second_pass() -> None:
                nonlocal failed
                round2 = _second_pass_notice(discovered, first_round)
                if not round2:
                    return
                http_urls = [u for u in round2 if _use_http_second_pass(u)]
                browser_urls = [u for u in round2 if u not in set(http_urls)]
                if verbose and http_urls:
                    self.stdout.write(
                        style.NOTICE(
                            f"Second pass: {len(http_urls)} URL(s) via parallel HTTP; "
                            f"{len(browser_urls)} via Chrome (if any)."
                        )
                    )
                if progress_fetch and http_urls:
                    self.stdout.write(
                        style.NOTICE(
                            f"HTTP 2nd pass per-URL timeout: {http_pass2_timeout:.0f}s "
                            f"(override via --http-pass2-timeout or OBS_HTTP_PASS2_PER_URL_TIMEOUT_S)."
                        )
                    )
                http_d = max(0.04, min(0.35, float(delay) / 6.0))
                if http_urls:
                    _wraw = (os.environ.get("OBS_HTTP_PASS2_WORKERS") or "12").strip()
                    workers = min(16, max(1, int(_wraw) if _wraw.isdigit() else 12))
                    fetch_http = partial(
                        fetch_page_extract_http,
                        delay=http_d,
                        target_lang=target_lang,
                    )
                    fut_u: dict[Any, str] = {}
                    http_started: dict[str, float] = {}
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        for i, u in enumerate(http_urls, start=1):
                            _log_fetch_start("2nd-http", i, len(http_urls), u)
                            http_started[u] = time.monotonic()
                            fut = ex.submit(fetch_http, u)
                            fut_u[fut] = u
                        for fut in as_completed(fut_u):
                            u = fut_u[fut]
                            t0 = http_started.pop(u, time.monotonic())
                            try:
                                title, text, units = fut.result(
                                    timeout=http_pass2_timeout
                                )
                            except FuturesTimeoutError:
                                failed += 1
                                elapsed = time.monotonic() - t0
                                logger.warning(
                                    "HTTP 2nd pass timeout %.1fs (limit %.1fs): %s",
                                    elapsed,
                                    http_pass2_timeout,
                                    u,
                                )
                                _log_fetch_done(u, elapsed, False, "timeout")
                                if verbose:
                                    self.stdout.write(
                                        style.WARNING(
                                            f"Timeout (2nd pass HTTP, {elapsed:.1f}s): {u}"
                                        )
                                    )
                                continue
                            except Exception as exc:
                                failed += 1
                                elapsed = time.monotonic() - t0
                                _log_fetch_done(u, elapsed, False, "HTTP")
                                if verbose:
                                    logger.warning("HTTP 2nd pass failed %s: %s", u, exc)
                                    self.stdout.write(
                                        style.WARNING(
                                            f"Failed (2nd pass HTTP): {u} ({exc})"
                                        )
                                    )
                                else:
                                    logger.debug("HTTP 2nd pass failed %s: %s", u, exc)
                                continue
                            elapsed = time.monotonic() - t0
                            _log_fetch_done(u, elapsed, True, "HTTP")
                            _save_one(u, title, text, units)
                for j, page_url in enumerate(browser_urls, start=1):
                    _maybe_start_stall_monitor()
                    _begin_stall(page_url)
                    t0 = time.monotonic()
                    _log_fetch_start("2nd-selenium", j, len(browser_urls), page_url)
                    try:
                        title, text, units, _f = fetch_page_extract(
                            collect_driver,
                            page_url,
                            delay,
                            target_lang=target_lang,
                        )
                    except Exception as exc:
                        failed += 1
                        _log_fetch_done(page_url, time.monotonic() - t0, False)
                        if verbose:
                            logger.warning("Failed 2nd pass %s: %s", page_url, exc)
                            self.stdout.write(
                                style.WARNING(f"Failed (2nd pass): {page_url} ({exc})")
                            )
                        else:
                            logger.debug("Failed 2nd pass %s: %s", page_url, exc)
                        _end_stall()
                        continue
                    _log_fetch_done(page_url, time.monotonic() - t0, True)
                    _end_stall()
                    _save_one(page_url, title, text, units)

            first_round = set(urls)
            discovered: set[str] = set()
            disc_lock = threading.Lock()

            if fetch_workers == 1:
                _maybe_start_stall_monitor()
                for i, url in enumerate(urls, start=1):
                    _begin_stall(url)
                    t0 = time.monotonic()
                    _log_fetch_start("1st", i, len(urls), url)
                    try:
                        title, text, embedding_units, follows = fetch_page_extract(
                            collect_driver,
                            url,
                            delay,
                            target_lang=target_lang,
                        )
                        discovered.update(follows)
                    except Exception as exc:
                        failed += 1
                        _log_fetch_done(url, time.monotonic() - t0, False)
                        if verbose:
                            logger.warning("Failed to fetch %s: %s", url, exc)
                            self.stdout.write(style.WARNING(f"Failed: {url} ({exc})"))
                        else:
                            logger.debug("Failed to fetch %s: %s", url, exc)
                        _end_stall()
                        continue
                    _log_fetch_done(url, time.monotonic() - t0, True)
                    _end_stall()
                    _save_one(url, title, text, embedding_units)
            else:
                pool: queue.Queue[Any] = queue.Queue()
                pool.put(collect_driver)
                extra = fetch_workers - 1
                if verbose:
                    self.stdout.write(
                        style.NOTICE(
                            f"Parallel fetch: {fetch_workers} Chrome instance(s) "
                            f"({extra} extra after discovery driver)."
                        )
                    )
                    _out_flush()
                if extra > 0 and (verbose or progress_fetch):
                    self.stdout.write(
                        style.NOTICE(
                            f"Launching {extra} more browser(s) for parallel fetch — "
                            "each can take ~20–90s in Docker before the first fetch line appears."
                        )
                    )
                    _out_flush()
                for wi in range(extra):
                    if verbose or progress_fetch:
                        self.stdout.write(
                            style.NOTICE(f"  Starting browser {wi + 1}/{extra}…")
                        )
                        _out_flush()
                    d = build_driver()
                    drivers_to_quit.append(d)
                    pool.put(d)
                    if verbose or progress_fetch:
                        self.stdout.write(style.SUCCESS(f"  Browser {wi + 1}/{extra} ready."))
                        _out_flush()

                def _fetch_with_pooled_driver(
                    page_url: str,
                    idx: int,
                    total: int,
                ) -> tuple[str, str, str, list[str], list[str]]:
                    """Return (url, title, text, embedding_units, followups) — no DB in thread."""
                    t0 = time.monotonic()
                    _log_fetch_start("parallel", idx, total, page_url)
                    d = pool.get()
                    try:
                        title, text, units, follows = fetch_page_extract(
                            d,
                            page_url,
                            delay,
                            target_lang=target_lang,
                        )
                        _log_fetch_done(page_url, time.monotonic() - t0, True)
                        return page_url, title, text, units, follows
                    except Exception:
                        _log_fetch_done(page_url, time.monotonic() - t0, False)
                        raise
                    finally:
                        pool.put(d)

                fut_to_url: dict[Any, str] = {}
                with ThreadPoolExecutor(max_workers=fetch_workers) as ex:
                    for i, u in enumerate(urls, start=1):
                        fut = ex.submit(_fetch_with_pooled_driver, u, i, len(urls))
                        fut_to_url[fut] = u
                    for fut in as_completed(fut_to_url):
                        url = fut_to_url[fut]
                        try:
                            _, title, text, embedding_units, follows = fut.result()
                        except Exception as exc:
                            failed += 1
                            if verbose:
                                logger.warning("Failed to fetch %s: %s", url, exc)
                                self.stdout.write(style.WARNING(f"Failed: {url} ({exc})"))
                            else:
                                logger.debug("Failed to fetch %s: %s", url, exc)
                            continue
                        with disc_lock:
                            discovered.update(follows)
                        _save_one(url, title, text, embedding_units)

            _run_second_pass()

        finally:
            shutdown_event.set()
            for lg, prev in log_levels_restored:
                lg.setLevel(prev)
            _quit_all_drivers()

        skip_bf = bool(options.get("skip_prog_backfill"))
        if not dry_run and not skip_bf:
            _out_flush()
            self.stdout.write(
                style.NOTICE(
                    "Post-scrape: HTTP prog*.aspx backfill from showPac rows (closes common gaps)…",
                )
            )
            _out_flush()
            run_backfill_obs_prog_tabs_http(
                limit=0,
                force=False,
                dry_run=False,
                delay=max(0.04, min(0.12, delay / 6.0)),
                target_lang=target_lang,
                on_notice=lambda m: self.stdout.write(style.NOTICE(m)),
                on_saved=(
                    (lambda m: self.stdout.write(style.SUCCESS(m))) if verbose else None
                ),
                on_warn=lambda m: self.stderr.write(style.WARNING(m)),
            )
        elif not dry_run and skip_bf and verbose:
            self.stdout.write(
                style.WARNING(
                    "Skipped post-scrape prog*.aspx backfill (--skip-prog-backfill).",
                )
            )

        if verbose:
            self.stdout.write(
                style.NOTICE(
                    f"Done. URLs processed={len(urls)}, "
                    f"saved={saved if not dry_run else 0}, failed={failed}, dry_run={dry_run}."
                )
            )
