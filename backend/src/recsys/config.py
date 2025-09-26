import yaml
from typing import Any, Dict

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load project config once, from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
