#!/usr/bin/env python3
"""Scrape your swudb.com decks into the local decklist format.

Two stages:

  1. DISCOVER  (needs a logged-in session)
     Opens https://swudb.com/decks/ and collects every /deck/<id> link.
     Use --discover-only to just dump urls.txt and eyeball it.

  2. EXPORT  (works logged out for unlisted decks; private ones are skipped)
     For each URL: opens the deck page, clicks "Export deck" -> "Copy JSON",
     captures the JSON that gets copied, then writes:

         { "<metadata.name>": <deck json>, "...": <deck json>, ... }

Usage
-----
  python scrape_swudb.py                      # discover + export everything
  python scrape_swudb.py --discover-only      # only write urls.txt
  python scrape_swudb.py --urls-file urls.txt # skip discovery, export those
  python scrape_swudb.py --merge              # merge into an existing output
  python scrape_swudb.py --limit 3 --debug    # try a few decks first

Setup (once)
------------
  pip install playwright
  playwright install chromium

The first run opens a dedicated browser profile (default: .swudb_profile/).
Log in to SWUDB in that window; the session is reused on later runs. The
browser is Microsoft Edge by default (--browser msedge); pass
  --browser chrome    to use installed Chrome
  --browser ""        to use Playwright's bundled Chromium
To reuse an existing profile instead, close that browser first and pass
  --profile-dir "C:\\path\\to\\profile"
(Playwright refuses to open a profile that a running browser already holds.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "decklist_scrap.json"
DEFAULT_URLS = SCRIPT_DIR / "urls.txt"
DEFAULT_PROFILE = SCRIPT_DIR / ".swudb_profile"
DEFAULT_DECKS_PAGE = "https://swudb.com/decks/"

_DECK_URL_RE = re.compile(
    r"^https?://(?:www\.)?swudb\.com/deck/([A-Za-z0-9_-]+)(?:[/?#].*)?$", re.I
)
_EXPORT_RE = re.compile(r"export", re.I)
_COPY_JSON_RE = re.compile(r"copy\s*json", re.I)

# Injected before every page load: whatever the site copies (navigator.clipboard
# OR the legacy execCommand('copy') path) is stashed on window for us to read.
CAPTURE_SCRIPT = r"""
(() => {
  const stash = (t) => { if (t) { window.__swudb_last_copied = String(t); } };
  try {
    const clip = navigator.clipboard;
    if (clip) {
      const orig = clip.writeText ? clip.writeText.bind(clip) : null;
      const wrapped = function (text) { stash(text); return orig ? orig(text) : Promise.resolve(); };
      try { Object.defineProperty(clip, 'writeText', { value: wrapped, configurable: true, writable: true }); } catch (e) {}
      try {
        const proto = Object.getPrototypeOf(clip);
        if (proto && proto.writeText) { proto.writeText = wrapped; }
      } catch (e) {}
    }
  } catch (e) {}
  document.addEventListener('copy', (e) => {
    try { stash(e.clipboardData && e.clipboardData.getData('text/plain')); } catch (e) {}
  }, true);
})();
"""


class DeckExportError(RuntimeError):
    pass


# ── url helpers ──────────────────────────────────────────────────────────────

def normalise_deck_url(href: str | None) -> str | None:
    """'/deck/vaMkqfdbIK' or 'https://swudb.com/deck/xvoghAjbAqUJH?x=1'
    -> 'https://swudb.com/deck/vaMkqfdbIK'. Returns None for non-deck links."""
    if not href:
        return None
    href = href.strip()
    if href.startswith("/"):
        href = "https://swudb.com" + href
    match = _DECK_URL_RE.match(href)
    if not match:
        return None
    return f"https://swudb.com/deck/{match.group(1)}"


def read_urls(path: Path) -> list[str]:
    """Read URLs from a file: newline, comma or space separated, deduped."""
    text = path.read_text(encoding="utf-8")
    urls: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", text):
        url = normalise_deck_url(token)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def write_urls(path: Path, urls: list[str]) -> None:
    path.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")


# ── browser helpers ──────────────────────────────────────────────────────────

def first_visible(locator, timeout_ms: int):
    try:
        candidate = locator.first
        candidate.wait_for(state="visible", timeout=timeout_ms)
        return candidate
    except Exception:
        return None


def find_clickable(page, pattern: re.Pattern, timeout_ms: int):
    """Try the usual roles, then fall back to any element with that text."""
    for locator in (
        page.get_by_role("button", name=pattern),
        page.get_by_role("menuitem", name=pattern),
        page.get_by_role("link", name=pattern),
        page.get_by_text(pattern),
    ):
        hit = first_visible(locator, timeout_ms)
        if hit is not None:
            return hit
    return None


def describe_page(page, limit: int = 12) -> str:
    """Debug helper: list clickable texts on the page."""
    try:
        items = page.evaluate(
            """(limit) => Array.from(document.querySelectorAll('button, a, [role=button], [role=menuitem], li'))
                 .map(e => (e.innerText || e.textContent || '').trim())
                 .filter(t => t && t.length < 40)
                 .slice(0, limit)""",
            limit,
        )
        return " | ".join(items) if items else "(no clickable text found)"
    except Exception as exc:  # pragma: no cover - debug path
        return f"(failed to inspect page: {exc})"


def discover_deck_urls(page, decks_page: str, debug: bool = False) -> list[str]:
    """Scroll / click 'load more' on the decks page until no new links appear."""
    print(f"[discover] opening {decks_page}")
    page.goto(decks_page, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2500)

    if any(word in page.url.lower() for word in ("login", "signin", "sign-in")):
        print("[discover] You are not logged in. Log in in the browser window, then press Enter here.")
        input()

    found: list[str] = []
    seen: set[str] = set()
    stale = 0
    for _ in range(300):
        try:
            hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.getAttribute('href'))")
        except Exception:
            hrefs = []

        new_links = 0
        for href in hrefs:
            url = normalise_deck_url(href)
            if url and url not in seen:
                seen.add(url)
                found.append(url)
                new_links += 1

        clicked = False
        for label in ("load more", "show more", "next"):
            button = first_visible(page.get_by_role("button", name=re.compile(label, re.I)), 600)
            if button is None:
                continue
            try:
                button.click()
                clicked = True
                page.wait_for_timeout(900)
                break
            except Exception:
                pass

        try:
            page.mouse.wheel(0, 30000)
        except Exception:
            pass
        page.wait_for_timeout(700)

        stale = 0 if (new_links or clicked) else stale + 1
        if stale >= 4:
            break

    if debug and not found:
        print(f"[discover] page text: {describe_page(page)}")
    print(f"[discover] found {len(found)} deck links")
    return found


def export_deck_json(page, url: str, debug: bool = False) -> dict:
    """Open a deck page, click Export deck -> Copy JSON, return the JSON."""
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.evaluate("window.__swudb_last_copied = null")
    page.wait_for_timeout(600)

    trigger = find_clickable(page, _EXPORT_RE, 8000)
    if trigger is None:  # page may still be hydrating
        page.wait_for_timeout(2000)
        trigger = find_clickable(page, _EXPORT_RE, 8000)
    if trigger is None:
        raise DeckExportError(
            "no 'Export' control found (private deck or page changed?)"
            + (f" — visible text: {describe_page(page)}" if debug else "")
        )

    # Some dropdowns only open on hover.
    try:
        trigger.hover()
        page.wait_for_timeout(150)
    except Exception:
        pass
    trigger.click()
    page.wait_for_timeout(300)

    item = find_clickable(page, _COPY_JSON_RE, 3000)
    if item is None:
        # retry via hover, then via a broader text match on the open menu
        try:
            trigger.hover()
            page.wait_for_timeout(300)
        except Exception:
            pass
        item = find_clickable(page, _COPY_JSON_RE, 3000)
    if item is None:
        raise DeckExportError(
            "no 'Copy JSON' item found after opening the export menu"
            + (f" — visible text: {describe_page(page)}" if debug else "")
        )

    item.click()
    page.wait_for_timeout(300)

    captured = None
    try:
        captured = page.evaluate("window.__swudb_last_copied")
    except Exception:
        captured = None
    if not captured:  # fallbacks: real clipboard read, then OS clipboard
        try:
            captured = page.evaluate("navigator.clipboard.readText()")
        except Exception:
            captured = None
    if not captured:
        try:
            import pyperclip  # optional
            captured = pyperclip.paste()
        except Exception:
            captured = None
    if not captured:
        raise DeckExportError("'Copy JSON' was clicked but nothing was captured")

    try:
        data = json.loads(captured)
    except json.JSONDecodeError as exc:
        raise DeckExportError(f"captured text is not JSON ({exc})") from exc
    if not isinstance(data, dict) or not any(key in data for key in ("deck", "leader", "base")):
        raise DeckExportError("captured JSON does not look like a deck export")

    return data


# ── output ───────────────────────────────────────────────────────────────────

def deck_key(deck: dict, url: str) -> str:
    metadata = deck.get("metadata") or {}
    return str(metadata.get("name") or "").strip() or url.rstrip("/").rsplit("/", 1)[-1]


def save(out_path: Path, decks: dict) -> None:
    ordered = {name: decks[name] for name in sorted(decks)}
    out_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


_BRANDED_PATHS = {
    "msedge": [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ],
    "chrome": [
        Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ],
}


def _branded_binary(channel: str) -> str | None:
    """Locate an installed Edge/Chrome binary when the Playwright channel
    lookup fails (non-standard install path, missing registry probe, ...)."""
    for candidate in _BRANDED_PATHS.get((channel or "").lower(), []):
        if candidate.exists():
            return str(candidate)
    return None


def launch_context(playwright, args):
    profile = Path(args.profile_dir).expanduser().resolve()
    profile.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        user_data_dir=str(profile),
        headless=args.headless,
        viewport={"width": 1400, "height": 900},
        permissions=["clipboard-read", "clipboard-write"],
        args=["--disable-blink-features=AutomationControlled"],
    )

    context = None
    if args.browser:
        try:
            context = playwright.chromium.launch_persistent_context(channel=args.browser, **kwargs)
        except Exception as exc:
            binary = _branded_binary(args.browser)
            if binary:
                print(f"[browser] channel '{args.browser}' failed ({exc})")
                print(f"[browser] trying installed binary: {binary}")
                try:
                    context = playwright.chromium.launch_persistent_context(executable_path=binary, **kwargs)
                except Exception as binary_exc:
                    print(f"[browser] {binary} failed ({binary_exc})")
            else:
                print(f"[browser] channel '{args.browser}' failed ({exc})")
            if context is None:
                print("[browser] falling back to Playwright's bundled Chromium")

    if context is None:
        context = playwright.chromium.launch_persistent_context(**kwargs)

    context.add_init_script(CAPTURE_SCRIPT)
    try:
        print(f"[browser] launched {context.browser.version}")
    except Exception:
        pass
    return context


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape swudb.com decks into the local decklist JSON format")
    parser.add_argument("--decks-page", default=DEFAULT_DECKS_PAGE, help="Deck list page to scrape links from")
    parser.add_argument("--urls", nargs="*", default=None, help="Explicit deck URLs (skips discovery)")
    parser.add_argument("--urls-file", type=Path, default=None, help="File with deck URLs (skips discovery)")
    parser.add_argument("--urls-out", type=Path, default=DEFAULT_URLS, help="Where discovery writes the URL list")
    parser.add_argument("--discover-only", action="store_true", help="Only collect URLs, do not open each deck")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output decklist JSON")
    parser.add_argument("--merge", action="store_true", help="Merge into an existing output file")
    parser.add_argument("--overwrite", action="store_true", help="With --merge, replace existing entries on name clash")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N decks")
    parser.add_argument("--delay", type=float, default=0.6, help="Seconds to wait between decks")
    parser.add_argument("--profile-dir", default=str(DEFAULT_PROFILE), help="Persistent browser profile directory")
    parser.add_argument("--browser", default="msedge", help="Browser channel ('chrome', 'msedge', or '' for Chromium)")
    parser.add_argument("--headless", action="store_true", help="Run without a window (cannot log in first time)")
    parser.add_argument("--debug", action="store_true", help="Print page details when a deck fails")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run:\n  pip install playwright\n  playwright install chromium")
        return 2

    explicit_urls: list[str] | None = None
    if args.urls:
        explicit_urls = [u for u in (normalise_deck_url(x) for x in args.urls) if u]
    elif args.urls_file:
        explicit_urls = read_urls(args.urls_file)
        print(f"[urls] loaded {len(explicit_urls)} deck urls from {args.urls_file}")

    decks: dict[str, dict] = {}
    if args.merge and args.out.exists():
        try:
            decks = json.loads(args.out.read_text(encoding="utf-8"))
            print(f"[out] merging into {args.out} ({len(decks)} existing decks)")
        except Exception:
            print(f"[out] could not parse {args.out}; starting fresh")

    with sync_playwright() as playwright:
        context = launch_context(playwright, args)
        page = context.new_page()

        if explicit_urls is None:
            urls = discover_deck_urls(page, args.decks_page, debug=args.debug)
            write_urls(args.urls_out, urls)
            print(f"[urls] wrote {len(urls)} urls to {args.urls_out}")
            if args.discover_only:
                context.close()
                return 0
        else:
            urls = explicit_urls

        if args.limit:
            urls = urls[: args.limit]
        if not urls:
            print("No deck urls to export.")
            context.close()
            return 1

        exported = skipped = 0
        for index, url in enumerate(urls, 1):
            print(f"[{index}/{len(urls)}] {url}")
            try:
                deck = export_deck_json(page, url, debug=args.debug)
            except Exception as exc:
                skipped += 1
                print(f"    skipped: {exc}")
                continue

            name = deck_key(deck, url)
            if name in decks and not args.overwrite:
                skipped += 1
                print(f"    skipped: a deck named {name!r} is already present (use --overwrite to replace)")
                continue
            decks[name] = deck
            exported += 1
            save(args.out, decks)  # save incrementally so a crash keeps progress
            page.wait_for_timeout(int(args.delay * 1000))

        print(f"\nWrote {len(decks)} decks to {args.out} (new/updated: {exported}, skipped: {skipped})")
        context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
