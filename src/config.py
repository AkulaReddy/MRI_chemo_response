"""YAML configs with single inheritance (`inherit: base.yaml`), deep-merged; plus a stable hash."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    return _merge(_load(path.parent / parent), cfg) if parent else cfg


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Config by path or by name in configs/ (e.g. `f2_attn` or `f2_attn.yaml`). `name` = file stem."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if not path.exists() and (CONFIG_DIR / path.name).exists():
        path = CONFIG_DIR / path.name
    cfg = _load(path)
    cfg["name"] = path.stem
    return _merge(cfg, overrides or {})


def set_dotted(cfg: dict, key: str, value):
    """`--set optim.max_epochs=2` style overrides; value parsed as YAML."""
    d = cfg
    *head, last = key.split(".")
    for k in head:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(value) if isinstance(value, str) else value


def config_hash(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
