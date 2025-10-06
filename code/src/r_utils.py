import re
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_R_VERSION = "4.5.1"


def get_r_version(rscript: str) -> str | None:
    """Return 'X.Y.Z' reported by `Rscript --version`, or None if unavailable."""
    try:
        out = subprocess.check_output([rscript, "--version"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    m = re.search(r"\b(\d+\.\d+\.\d+)\b", out)
    return m.group(1) if m else None


def detect_rscript() -> str:
    """Locate an Rscript matching REQUIRED_R_VERSION, warn if others are found."""
    wrong_versions: list[tuple[str, str]] = []
    checked: set[str] = set()
    found: str | None = None

    # Candidates: PATH first, then known locations
    candidates = [
        shutil.which("Rscript") or "",
        # Windows
        r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe",
        # macOS / Linux
        "/opt/homebrew/bin/Rscript",  # Homebrew (Apple Silicon)
        "/usr/local/bin/Rscript",
        "/usr/bin/Rscript",
        str(Path.home() / "R" / "4.5.1" / "bin" / "Rscript"),
    ]

    for path in candidates:
        if not path or path in checked or not Path(path).exists():
            continue
        checked.add(path)
        version = get_r_version(path)
        if not version:
            continue
        if version == REQUIRED_R_VERSION and not found:
            found = path
        elif version != REQUIRED_R_VERSION:
            wrong_versions.append((path, version))

    if found:
        print(f"✓ Using Rscript: {found} (version {REQUIRED_R_VERSION})")
        if wrong_versions:
            shown = ", ".join(f"{v} at {p}" for p, v in wrong_versions[:2])
            extra = f", +{len(wrong_versions) - 2} more" if len(wrong_versions) > 2 else ""
            print(f"⚠️  Also detected other versions: {shown}{extra}")
        return found

    # Fail cleanly
    print(f"✗ Could not find Rscript {REQUIRED_R_VERSION}.", file=sys.stderr)
    if wrong_versions:
        for p, v in wrong_versions:
            print(f"  Found Rscript {v} at {p}", file=sys.stderr)
    print(
        "Please install R 4.5.1 and ensure Rscript is on PATH, or update the known paths in this script.",
        file=sys.stderr,
    )
    sys.exit(1)
