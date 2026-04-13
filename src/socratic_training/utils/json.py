from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Union


def extract_first_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Best-effort extraction of the first JSON object/array embedded in model text.
    """
    # Try array first, then object.
    for open_c, close_c in [("[", "]"), ("{", "}")]:
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start == -1 or end == -1 or end <= start:
            continue
        blob = text[start : end + 1]
        try:
            return json.loads(blob)
        except Exception:
            continue
    raise ValueError("No valid JSON object/array found in text.")

