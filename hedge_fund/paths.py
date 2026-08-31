"""Where user data lives: ~/.hedge-fund/.

Everything the user owns — mandates, run/backtest receipts, API caches, and
the .env key file — lives under one home directory, outside the package. The
package directory stays read-only code, so a pipx install behaves exactly
like a checkout.

Textual-free and import-light on purpose: every layer (CLI, TUI, caches)
anchors its paths here, and nothing here may import them back.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_user_dir(value: str | None = None) -> Path:
    """Resolve the application data directory.

    ``AGENTIC_QUANT_HOME`` makes runs isolated and reproducible in CI,
    containers, and research sandboxes. Existing installs keep using the
    legacy directory when the variable is unset.
    """
    configured = value if value is not None else os.environ.get("AGENTIC_QUANT_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".hedge-fund"


USER_DIR = resolve_user_dir()
MANDATES_DIR = USER_DIR / "mandates"
CACHE_DIR = USER_DIR / "cache"
ENV_PATH = USER_DIR / ".env"

# The example mandate ships inside the package; it is copied out (never read
# in place) so users edit their copy, not the install.
EXAMPLE_MANDATE = Path(__file__).resolve().parent / "fund" / "example.yaml"


def ensure_mandates_dir() -> Path:
    """Create the mandates dir on first use, seeded with the example."""
    if not MANDATES_DIR.exists():
        MANDATES_DIR.mkdir(parents=True)
        shutil.copy(EXAMPLE_MANDATE, MANDATES_DIR / "example.yaml")
    return MANDATES_DIR
