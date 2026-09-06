"""Token refresh sidecar — launches browser, extracts JWT, writes token file."""

from __future__ import annotations

import fcntl
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.auth import (  # noqa: E402
    REFRESH_LOCK_PATH,
    get_token_expiry,
    write_token_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

LOCK_PATH = REFRESH_LOCK_PATH
MFA_TIMEOUT_MS = 300_000

_SSO_SELECTORS = (
    'text="Continue with University of Arizona"',
    'button:has-text("Continue with University of Arizona")',
    'a:has-text("Continue with University of Arizona")',
)

_OTHER_OPTIONS_SELECTORS = (
    'a:has-text("Other options")',
    'button:has-text("Other options")',
    'text="Other options"',
)

_PUSH_SELECTORS = (
    '[data-testid="test-auth-method-duo_push"]',
    'button:has-text("Duo Push")',
    'a:has-text("Duo Push")',
    'text="Send me a Push"',
)

_TRUST_SELECTORS = (
    '#trust-browser-button',
    'button:has-text("Yes, this is my device")',
    'button:has-text("Yes, trust browser")',
)


def _any_locator(page, selectors: tuple[str, ...]):
    """Build one locator matching any of the selectors.

    Comma-joining these into a single selector string does not work: Playwright cannot
    parse a list that mixes engines (`text="..."` with `:has-text(...)`), and commas
    inside the quoted text break the grammar. `or_()` is the supported way to race them.
    """
    loc = page.locator(selectors[0])
    for selector in selectors[1:]:
        loc = loc.or_(page.locator(selector))
    return loc


def _click_first(page, selectors: tuple[str, ...], timeout: int = 0) -> str | None:
    """Click the first matching selector.

    A non-zero timeout is spent once on the whole group, not per selector — waiting
    serially would multiply the timeout by the number of selectors when none match.
    """
    if timeout:
        try:
            _any_locator(page, selectors).first.wait_for(timeout=timeout, state="visible")
        except Exception:
            return None
    for selector in selectors:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                el.click()
                return selector
        except Exception:
            continue
    return None


def _prefer_duo_push(page) -> None:
    """Switch Duo from code-entry mode to tap-to-approve Push, which needs no typed code."""
    if _click_first(page, _PUSH_SELECTORS):
        logger.info("Duo Push already offered — selected it")
        time.sleep(2)
        return

    if _click_first(page, _OTHER_OPTIONS_SELECTORS):
        logger.info("Opened Duo 'Other options'")
        time.sleep(2)
        hit = _click_first(page, _PUSH_SELECTORS)
        if hit:
            logger.info("Selected Duo Push via %s", hit)
            time.sleep(2)
            return
        logger.warning("Duo Push not listed under 'Other options'")
    else:
        logger.info("No 'Other options' link found — staying on current Duo method")


def _log_duo_instructions(page) -> None:
    try:
        text = " ".join(page.inner_text("body").split())
    except Exception:
        return
    logger.info("Duo page state: %s", text[:300])
    if "Enter code" in text or "verification code" in text:
        logger.warning(
            "Duo is in CODE-ENTRY mode — this cannot complete unattended. "
            "Type the displayed code into Duo Mobile, or enable Duo Push for this account."
        )


def _accept_trust_prompt(page) -> None:
    """Duo's 'trust this browser' prompt appears only after approval; clicking it persists the session."""
    hit = _click_first(page, _TRUST_SELECTORS)
    if hit:
        logger.info("Accepted 'trust this browser' via %s", hit)


def _acquire_lock() -> int | None:
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_WRONLY, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        logger.info("Refresh already in progress (lock held)")
        return None
    except OSError as exc:
        logger.error("Lock error: %s", exc)
        return None


def _release_lock(fd: int) -> None:
    # Never unlink the lock path: a fresh inode would let a concurrent caller
    # acquire immediately, defeating mutual exclusion.
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _write_token_file(token: str, expires_at: int | None) -> None:
    token_path = write_token_file(token, expires_at)
    logger.info("Token written to %s (expires_at=%s)", token_path, expires_at)


def run() -> bool:
    lock_fd = _acquire_lock()
    if lock_fd is None:
        return False
    try:
        return _run_with_lock(lock_fd)
    finally:
        _release_lock(lock_fd)


def _run_with_lock(_lock_fd: int) -> bool:
    from playwright.sync_api import sync_playwright

    url = os.environ["OPEN_WEBUI_URL"].rstrip("/")
    netid = os.environ.get("UA_NETID", "")
    password = os.environ.get("UA_NETID_PASSWORD", "")
    default_profile = str(Path.home() / ".config" / "open-webui-proxy" / "browser-profile")
    profile_dir = Path(os.environ.get("BROWSER_PROFILE_DIR", default_profile))

    logger.info("Starting browser login for %s", url)

    try:
        with sync_playwright() as pw:
            headless = os.environ.get("PLAYWRIGHT_HEADLESS", "true").lower() in ("true", "1", "yes")
            browser = pw.chromium.launch_persistent_context(
                str(profile_dir),
                headless=headless,
                ignore_https_errors=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            page = browser.pages[0] if browser.pages else browser.new_page()

            page.goto(url, wait_until="networkidle", timeout=60000)
            # Open WebUI is a client-rendered SPA that redirects to /auth after load, so
            # networkidle alone does not mean the SSO button exists yet. Wait for either
            # landmark before probing; a cold browser profile is markedly slower.
            try:
                _any_locator(page, (*_SSO_SELECTORS, "#username")).first.wait_for(
                    timeout=30000, state="visible"
                )
            except Exception:
                logger.info("No SSO button or login form appeared; may already be signed in")
            logger.info("Landed at: %s", page.url)

            cookies = page.context.cookies([url])
            has_token = any(c["name"] == "token" for c in cookies)

            if not has_token and netid and password:
                if not page.query_selector("#username"):
                    sso = _click_first(page, _SSO_SELECTORS, timeout=15000)
                    if sso:
                        logger.info("Clicked SSO entry point via %s", sso)
                    else:
                        logger.info("No SSO button found; expecting login form directly")

                logger.info("Waiting for Shibboleth login form")
                page.wait_for_selector("#username", timeout=60000)
                logger.info("Shibboleth login form found: %s", page.url)
                page.fill("#username", netid)
                page.fill("#password", password)
                page.get_by_role("button", name="Login").click()
                logger.info("Submitted credentials, waiting for navigation")

                time.sleep(3)
                logger.info("Post-login URL: %s", page.url)

                if "duosecurity.com" in page.url:
                    _prefer_duo_push(page)
                    _log_duo_instructions(page)

                logger.info("Waiting for Duo approval and token cookie")

                token = None
                start = time.time()
                while time.time() - start < MFA_TIMEOUT_MS / 1000:
                    toks = [c["value"] for c in page.context.cookies([url]) if c["name"] == "token"]
                    if toks:
                        token = toks[0]
                        break
                    _accept_trust_prompt(page)
                    time.sleep(2)
                    logger.info(
                        "Still waiting... (elapsed: %.0fs, url=%s)",
                        time.time() - start, page.url.split("?")[0],
                    )

                if not token:
                    logger.error("Timeout waiting for Duo approval")
                    browser.close()
                    return False

                logger.info("Token cookie found")
            else:
                logger.info("Extracting existing token from browser cookies")
                token = None
                for c in cookies:
                    if c["name"] == "token":
                        token = c["value"]
                        break
                if not token:
                    logger.error("No token cookie found")
                    browser.close()
                    return False

            expires_at = get_token_expiry(token)
            if expires_at is None:
                logger.warning("Could not decode token expiry")

            _write_token_file(token, expires_at)
            browser.close()
            logger.info("Token refresh complete")
            return True

    except Exception as exc:
        logger.exception("Sidecar failure: %s", exc)
        return False


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
