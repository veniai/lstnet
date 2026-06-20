"""Package-level path resolution and configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """Return the resolved directory containing the ``lstnet`` package.

    Resolves to the ``src/lstnet`` directory in a source checkout and to the
    installed package directory in an installed environment.
    """
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Return the resolved repository root (CWD-independent).

    ``package_root()`` is ``src/lstnet``; two ``.parent`` calls walk up to
    ``src/`` then the repo root. Used as the anchor for default ``data_dir``
    paths so readers do not depend on the current working directory.
    """
    return package_root().parent.parent


def earthdata_credentials() -> tuple[str, str]:
    """Return NASA Earthdata Login ``(username, password)`` from the environment.

    Reads ``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD``. Credentials must
    NEVER be hardcoded — the legacy app stored them in plain source, which is a
    defect this refactor removes. Raise ``RuntimeError`` with setup guidance if
    either is missing.

    (``earthaccess`` also honours ``~/.netrc`` / its own persisted login; this
    helper is the explicit, documented path for programmatic use.)
    """
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "NASA Earthdata credentials not found. Set EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD in the environment (or configure ~/.netrc)."
        )
    return username, password

