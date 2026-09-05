"""Renders OpenAI-style chat messages into a single prompt string via the
tokenizer's chat template, so /v1/chat/completions works for any instruct model."""

from transformers import AutoTokenizer

from backend.core.schemas import ChatMessage

_tokenizer_cache: dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def render_chat_prompt(model_name: str, messages: list[ChatMessage]) -> str:
    tokenizer = _get_tokenizer(model_name)
    formatted = [{"role": m.role, "content": m.content} for m in messages]
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            formatted, tokenize=False, add_generation_prompt=True
        )
    # Fallback for base models without a chat template.
    lines = [f"{m['role']}: {m['content']}" for m in formatted]
    lines.append("assistant:")
    return "\n".join(lines)
