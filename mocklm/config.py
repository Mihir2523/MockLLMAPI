"""
MockLLM Configuration
~~~~~~~~~~~~~~~~~~~~~
Global constants and default settings.
"""

import os
from pathlib import Path

# ── Browser Data ──────────────────────────────────────────────────────────────
# Persistent browser profile directory — stores cookies/sessions so user
# only needs to log in once per provider.
BROWSER_DATA_BASE = Path(os.path.expanduser("~")) / ".mocklm" / "browser_data"

# ── Timeouts (seconds) ───────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 120          # Max wait for a full LLM response
NAVIGATION_TIMEOUT = 30        # Max wait for page navigation
SELECTOR_TIMEOUT = 10          # Max wait to find a DOM element
STREAM_IDLE_TIMEOUT = 3.0      # If DOM text stops changing for this long, assume done
STREAM_POLL_INTERVAL = 0.5     # How often to check if streaming is still going

# ── Human-like Typing ─────────────────────────────────────────────────────────
TYPING_DELAY_MIN = 20          # Min milliseconds between keystrokes
TYPING_DELAY_MAX = 60          # Max milliseconds between keystrokes
PRE_CLICK_DELAY = 0.1          # Seconds to pause before clicking
POST_SUBMIT_DELAY = 1.0        # Seconds to wait after submitting before monitoring

# ── Browser Launch Options ────────────────────────────────────────────────────
HEADLESS_DEFAULT = False        # False = visible browser (needed for first login)
BROWSER_ARGS = [
    "--disable-blink-features=AutomationControlled",   # Hide automation flag
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-infobars",
]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PREFIX = "[MockLLM]"
LOG_LEVEL = "INFO"
