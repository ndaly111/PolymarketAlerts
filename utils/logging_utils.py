from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_logging(
    *,
    name: str = "",
    log_dir: Optional[Path] = None,
    debug: bool = False,
) -> Path:
    """
    Standard logging config for the entire repo.

    Priority:
      1) explicit debug=True
      2) LOG_LEVEL env
      3) DEBUG env
      4) default INFO
    """
    level_name = os.getenv("LOG_LEVEL", "").strip().upper()
    if not level_name:
        level_name = "DEBUG" if (debug or _env_truthy("DEBUG")) else "INFO"
    level = getattr(logging, level_name, logging.INFO)

    if log_dir is None:
        log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = (name or "run").replace("/", "_").replace(" ", "_")
    log_path = log_dir / f"{safe_name}_{ts}.log"

    # Avoid duplicate handlers if a script imports multiple modules that call configure_logging
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    root.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging configured level=%s file=%s", level_name, log_path)
    return log_path
