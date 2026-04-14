from __future__ import annotations

import io
import tokenize


def strip_python_comments(code: str) -> str:
    """
    Removes `# ...` comments from Python source code.

    Notes:
    - Preserves newlines/indentation as best-effort.
    - Does NOT remove `#` inside strings.
    - If tokenization fails, returns the input unchanged.
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        out = []
        for tok_type, tok_str, _start, _end, _line in tokens:
            if tok_type == tokenize.COMMENT:
                continue
            out.append((tok_type, tok_str))
        return tokenize.untokenize(out)
    except Exception:
        return code

