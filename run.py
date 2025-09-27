# run.py
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any

def _load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".yaml", ".yml"]:
        try:
            import yaml  # PyYAML
        except ImportError:
            raise SystemExit("PyYAML not installed. Run: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise SystemExit(f"Unsupported config extension '{ext}'. Use .yaml/.yml or .json")

def main():
    parser = argparse.ArgumentParser(description="Config-driven training runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config .yaml/.yml/.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Lazy import to avoid heavy deps on simple --help
    from src.model import run
    summaries = run(cfg)
    # Pretty-print final results
    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
