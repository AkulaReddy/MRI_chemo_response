import sys as _sys


def hide_broken_torchaudio():
    """transformers imports torchaudio when it is installed. A torchaudio built for another torch
    (e.g. on the project laptop: torchaudio 2.11 with torch 2.6) fails with OSError and breaks every
    model import. We never use audio, so mark it unavailable instead. Call it before importing
    transformers; it imports torch, so `import src` alone stays cheap (clinical-only code)."""
    try:
        import torchaudio  # noqa: F401
    except Exception:
        for name in [m for m in _sys.modules if m == "torchaudio" or m.startswith("torchaudio.")]:
            del _sys.modules[name]
        _sys.modules["torchaudio"] = None

