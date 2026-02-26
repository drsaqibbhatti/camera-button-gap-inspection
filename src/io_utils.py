from __future__ import annotations
from pathlib import Path
import json
import yaml

def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path: str, obj: dict):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
