import os
import sys
from pathlib import Path


def get_cache_dir() -> Path:
    if sys.platform == "win32":
        # Use the LOCALAPPDATA environment variable on Windows.
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            cache_dir = Path(local_appdata) / ".cache"
        else:
            cache_dir = Path.home() / ".cache"
    elif sys.platform == "darwin":
        # Use the macOS default cache directory.
        cache_dir = Path.home() / "Library" / "Caches"
    else:
        # On Linux/Unix, check XDG_CACHE_HOME; fallback to ~/.cache.
        xdg_cache = os.getenv("XDG_CACHE_HOME")
        if xdg_cache:
            cache_dir = Path(xdg_cache)
        else:
            cache_dir = Path.home() / ".cache"

    # Create the cache directory if it does not exist.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
