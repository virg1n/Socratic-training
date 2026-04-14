from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Union


def extract_first_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Best-effort extraction of the first JSON object/array embedded in model text.
    """
    dec = json.JSONDecoder()
    # Strip optional chain-of-thought blocks used by some models (Red/Judge may emit these).
    # This makes extraction robust when the <think> block contains braces.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    # Normalize common fullwidth punctuation produced by some models.
    s = (
        text.strip()
        .translate(
            {
                ord("［"): "[",
                ord("］"): "]",
                ord("｛"): "{",
                ord("｝"): "}",
                ord("【"): "[",
                ord("】"): "]",
                ord("“"): '"',
                ord("”"): '"',
                ord("，"): ",",
                ord("："): ":",
            }
        )
    )
    for i, ch in enumerate(s):
        if ch not in "[{":
            continue
        try:
            obj, _end = dec.raw_decode(s[i:])
            return obj
        except Exception:
            continue
    raise ValueError("No valid JSON object/array found in text.")
