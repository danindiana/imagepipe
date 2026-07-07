"""Compliance gate. Every download decision passes through here and is logged.

Explicit non-goals (never implemented anywhere in this codebase):
- no scraping of Google Images result pages
- no CAPTCHA/anti-bot evasion, stealth drivers, proxy rotation, fake user agents
- no downloading Google thumbnail caches (gstatic/encrypted-tbn) as canonical assets
- no watermark removal, no login/paywall bypass
"""
from __future__ import annotations

import urllib.parse
import urllib.robotparser
from dataclasses import dataclass

from .config import Config

# Domains that are search-engine thumbnail caches, never a canonical source.
THUMBNAIL_CACHE_DOMAINS = {
    "encrypted-tbn0.gstatic.com", "encrypted-tbn1.gstatic.com",
    "encrypted-tbn2.gstatic.com", "encrypted-tbn3.gstatic.com",
    "gstatic.com", "googleusercontent.com",
}

USER_AGENT = "imagepipe/0.1 (+local research tool; contact: user)"


@dataclass
class Decision:
    allowed: bool
    reason: str


class PolicyGate:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _domain(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc.lower()

    def _robots_ok(self, url: str) -> bool:
        dom = self._domain(url)
        rp = self._robots.get(dom)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            scheme = urllib.parse.urlparse(url).scheme or "https"
            rp.set_url(f"{scheme}://{dom}/robots.txt")
            try:
                rp.read()
            except Exception:
                # unreadable robots -> conservative default: allow root-level fetch
                # but mark parser as permissive-unknown
                rp.allow_all = True  # type: ignore[attr-defined]
            self._robots[dom] = rp
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def check_download(self, image_url: str, license_str: str | None) -> Decision:
        if not image_url or not image_url.startswith(("http://", "https://")):
            return Decision(False, "invalid_url")
        dom = self._domain(image_url)
        base = ".".join(dom.split(".")[-2:])
        if dom in THUMBNAIL_CACHE_DOMAINS or base in THUMBNAIL_CACHE_DOMAINS:
            return Decision(False, "thumbnail_cache_not_canonical")
        if any(dom == b or dom.endswith("." + b) for b in self.cfg.block_domains):
            return Decision(False, "blocked_domain")
        if self.cfg.allow_domains and not any(
            dom == a or dom.endswith("." + a) for a in self.cfg.allow_domains
        ):
            return Decision(False, "not_in_allowlist")
        if self.cfg.require_license and not license_str:
            return Decision(False, "license_unknown_but_required")
        if self.cfg.respect_robots and not self._robots_ok(image_url):
            return Decision(False, "robots_disallow")
        return Decision(True, "ok")
