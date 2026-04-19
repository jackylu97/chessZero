"""Unique run ID generation for training runs.

Format: YYYY_MM_DD_NNNN — today's date plus a 4-digit daily sequence number.
Sequence is picked by scanning existing subdirectories under the provided
scan roots and choosing max(seq)+1 for today's prefix.
"""

from datetime import date
from pathlib import Path


def generate_run_id(*scan_dirs: Path) -> str:
    today = date.today().strftime("%Y_%m_%d")
    prefix = today + "_"

    existing = set()
    for root in scan_dirs:
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir() or not entry.name.startswith(prefix):
                continue
            try:
                existing.add(int(entry.name[len(prefix):]))
            except ValueError:
                continue

    return f"{today}_{max(existing, default=0) + 1:04d}"
