from __future__ import annotations

from typing import Any, Dict, Optional


def build_model_inputs(tokenizer: Any, *, user_text: str, system_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Tokenizes a prompt for chat/instruct models when a chat template is available.

    Falls back to plain text tokenization if no template exists.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for model inputs.") from e

    chat_template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply) and chat_template:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        encoded = apply(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if isinstance(encoded, dict):
            return encoded
        # Some tokenizers return just a tensor.
        return {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}

    text = f"{system_text}\n\n{user_text}" if system_text else user_text
    return tokenizer(text, return_tensors="pt")

