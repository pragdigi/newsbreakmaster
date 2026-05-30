"""Unit tests for ``_build_newsbreak_platforms`` (NewsBreak inventory picker).

Covers the validator rules from the official docs:
  https://advertising-api.newsbreak.com/hc/en-us/articles/45950972191629-Platforms

Goals:
  - Default mode (no opt-in) MUST omit the field so the API keeps the
    pre-Apr-2026 "Unlimited" behavior. We never want a stealth payload
    change for users who didn't touch the picker.
  - Custom mode honors NEWSBREAK + SCOOPZ checkboxes plus the single
    PREMIUM_PARTNERS_* radio.
  - The picker UI guards the API rules client-side, but the server still
    has to defend against tampered POSTs (custom HTML, replayed forms,
    automation scripts), so we re-enforce them here too.
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Iterable, List

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Form:
    """Minimal stand-in for Flask/Werkzeug ``request.form``.

    Supports both single-value ``.get`` and multi-value ``.getlist``,
    which is all ``_build_newsbreak_platforms`` needs.
    """

    def __init__(self, single: dict[str, str] | None = None,
                 multi: dict[str, Iterable[str]] | None = None):
        self._single = dict(single or {})
        self._multi = {k: list(v) for k, v in (multi or {}).items()}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._single.get(key, default)

    def getlist(self, key: str) -> List[str]:
        return list(self._multi.get(key, []))


class BuildNewsbreakPlatformsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing app.py boots a lot of optional Flask infra
        # (scheduler, storage init). We import lazily here so a missing
        # env var on dev machines doesn't break the whole test module.
        import app

        # Wrap as staticmethod so the class binding doesn't try to inject
        # `self` as the first positional argument when callers do
        # ``self._build(form)``.
        cls._build = staticmethod(app._build_newsbreak_platforms)

    # --- default behavior ----------------------------------------------

    def test_omit_when_mode_missing(self):
        # Old form posts (UI not redeployed yet) must keep working.
        self.assertIsNone(self._build(_Form()))

    def test_omit_when_mode_all(self):
        self.assertIsNone(self._build(_Form(single={"platforms_mode": "all"})))

    def test_mode_is_case_insensitive(self):
        self.assertIsNone(self._build(_Form(single={"platforms_mode": "ALL"})))

    # --- custom selection ----------------------------------------------

    def test_custom_newsbreak_only(self):
        form = _Form(
            single={"platforms_mode": "custom"},
            multi={"platforms": ["NEWSBREAK"]},
        )
        self.assertEqual(self._build(form), ["NEWSBREAK"])

    def test_custom_newsbreak_plus_scoopz(self):
        form = _Form(
            single={"platforms_mode": "custom"},
            multi={"platforms": ["NEWSBREAK", "SCOOPZ"]},
        )
        self.assertEqual(self._build(form), ["NEWSBREAK", "SCOOPZ"])

    def test_custom_with_premium_radio(self):
        form = _Form(
            single={
                "platforms_mode": "custom",
                "platforms_premium": "PREMIUM_PARTNERS_GAMING",
            },
            multi={"platforms": ["NEWSBREAK"]},
        )
        self.assertEqual(
            self._build(form),
            ["NEWSBREAK", "PREMIUM_PARTNERS_GAMING"],
        )

    def test_premium_only_no_checkboxes(self):
        # Some advertisers may want premium-partner inventory only.
        form = _Form(
            single={
                "platforms_mode": "custom",
                "platforms_premium": "PREMIUM_PARTNERS_ALL",
            },
            multi={"platforms": []},
        )
        self.assertEqual(self._build(form), ["PREMIUM_PARTNERS_ALL"])

    # --- safety / sanitization ------------------------------------------

    def test_dedupes_repeated_checkboxes(self):
        form = _Form(
            single={"platforms_mode": "custom"},
            multi={"platforms": ["NEWSBREAK", "NEWSBREAK", "SCOOPZ"]},
        )
        self.assertEqual(self._build(form), ["NEWSBREAK", "SCOOPZ"])

    def test_drops_unknown_values(self):
        # Tampered form posts must not leak unsupported values to Meta-side.
        form = _Form(
            single={"platforms_mode": "custom"},
            multi={"platforms": ["NEWSBREAK", "evil-source", ""]},
        )
        self.assertEqual(self._build(form), ["NEWSBREAK"])

    def test_strips_app_and_web_unlimited_from_list(self):
        # APP_AND_WEB_UNLIMITED must appear alone — easiest enforcement is
        # to never let it co-exist with explicit choices in the first place.
        form = _Form(
            single={"platforms_mode": "custom"},
            multi={"platforms": ["NEWSBREAK", "APP_AND_WEB_UNLIMITED"]},
        )
        self.assertEqual(self._build(form), ["NEWSBREAK"])

    def test_strips_premium_value_from_checkbox_list(self):
        # Premium values are a single-select via the radio; if a tampered
        # POST sends them as a checkbox, we drop them rather than
        # potentially producing a list with two PREMIUM_* entries.
        form = _Form(
            single={
                "platforms_mode": "custom",
                "platforms_premium": "PREMIUM_PARTNERS_NON_GAMING",
            },
            multi={
                "platforms": ["NEWSBREAK", "PREMIUM_PARTNERS_GAMING"],
            },
        )
        # Result keeps NEWSBREAK and the radio-chosen premium value, never
        # both PREMIUM_* entries.
        self.assertEqual(
            self._build(form),
            ["NEWSBREAK", "PREMIUM_PARTNERS_NON_GAMING"],
        )

    def test_empty_custom_falls_back_to_default(self):
        # User opened the panel, deselected everything, then submitted —
        # honoring "" as APP_AND_WEB_UNLIMITED is friendlier than a 400.
        form = _Form(
            single={"platforms_mode": "custom", "platforms_premium": ""},
            multi={"platforms": []},
        )
        self.assertIsNone(self._build(form))

    def test_invalid_premium_radio_is_ignored(self):
        # If somebody POSTs an unknown premium enum (e.g. typo or
        # rebranded value), drop it; don't 500 the launch.
        form = _Form(
            single={
                "platforms_mode": "custom",
                "platforms_premium": "PREMIUM_PARTNERS_NEW_THING",
            },
            multi={"platforms": ["NEWSBREAK"]},
        )
        self.assertEqual(self._build(form), ["NEWSBREAK"])


if __name__ == "__main__":
    unittest.main()
