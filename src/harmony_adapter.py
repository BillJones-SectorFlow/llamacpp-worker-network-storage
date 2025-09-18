# src/harmony_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Harmony → OpenAI Chat Completions adapter
#
# What this does:
# - For full (non-streaming) chat completions:
#     * Extracts "final" user-facing text into message.content (str)
#     * Extracts "analysis"/reasoning into message.reasoning_content (str) [LiteLLM-compatible]
#     * Handles providers that return message.reasoning as str/dict/list (OpenRouter-style)
#     * No-ops for responses that are already plain OpenAI format
# - For streaming chunks:
#     * Maps Harmony 'analysis' to delta.reasoning_content
#     * Maps Harmony 'final' (or plain content) to delta.content
#     * Passes through tool_calls, function calls, usage etc. untouched
#
# It is resilient to multiple provider variants:
# - content can be str, dict, or list[dict]
# - reasoning can be in message["reasoning"] or encoded by channel in content parts
#
# Optional controls via env (read/checked by caller):
# - include_reasoning (bool) -> whether to surface reasoning_content fields
# -----------------------------------------------------------------------------


def translate_chat_completion(obj: Dict[str, Any], include_reasoning: bool = True) -> Dict[str, Any]:
    """Normalize a non-streaming Chat Completions response in-place and return it."""
    if not isinstance(obj, dict) or obj.get("object") != "chat.completion":
        return obj

    choices = obj.get("choices") or []
    for ch in choices:
        msg = ch.get("message") or {}
        if not isinstance(msg, dict):
            continue

        # 1) provider-style explicit reasoning -> map to reasoning_content
        reasoning_text = _extract_reasoning_field(msg.get("reasoning"))
        # 2) content field might itself be Harmony parts -> split into final vs analysis
        content = msg.get("content")
        fin_from_parts, ana_from_parts = _split_harmony_content(content)

        # Decide final content
        if fin_from_parts is not None:
            msg["content"] = fin_from_parts
        else:
            # if content is list/dict weirdness, flatten to text
            if isinstance(content, list):
                msg["content"] = _join_texts(content)
            elif isinstance(content, dict):
                msg["content"] = _extract_text(content)
            # else: already a string -> leave as-is

        # Surface reasoning if requested
        mix_reasoning = _coalesce_texts(reasoning_text, ana_from_parts)
        if include_reasoning and mix_reasoning:
            msg["reasoning_content"] = mix_reasoning

        # Do not remove provider-specific 'reasoning' field; just leave it for parity
        ch["message"] = msg

    return obj


def translate_chat_chunk(chunk: Dict[str, Any], include_reasoning: bool = True) -> Dict[str, Any]:
    """Normalize a streaming Chat Completion chunk in-place and return it."""
    if not isinstance(chunk, dict) or chunk.get("object") != "chat.completion.chunk":
        return chunk

    for ch in (chunk.get("choices") or []):
        delta = ch.get("delta") or {}
        if not isinstance(delta, dict):
            continue

        # 1) explicit delta.reasoning (OpenRouter-style)
        reasoning_from_field = _extract_reasoning_field(delta.get("reasoning"))

        # 2) content may be string (plain), dict, or list of Harmony parts
        content = delta.get("content")
        fin_from_parts, ana_from_parts = _split_harmony_content(content)

        # If content was Harmony parts (list/dict), replace with normalized string
        if isinstance(content, list) or isinstance(content, dict):
            delta["content"] = fin_from_parts  # may be None for pure analysis turns

        # Add LiteLLM-compatible reasoning_content for analysis
        combined_reasoning = _coalesce_texts(reasoning_from_field, ana_from_parts)
        if include_reasoning and combined_reasoning:
            delta["reasoning_content"] = combined_reasoning

        ch["delta"] = delta

    return chunk


# ------------------------- helpers -------------------------

def _split_harmony_content(content: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (final_text, analysis_text) extracted from 'content'.
    - If content is a list[dict], look for parts where:
        part.get('channel') in {'final', 'analysis'}
        and text in part['text'] / part['content']
    - If content is a single dict, interpret similarly
    - If content is already a string, return (None, None) to indicate "no rewrite needed"
    """
    if isinstance(content, list):
        finals: List[str] = []
        analyses: List[str] = []
        for p in content:
            txt = _extract_text(p)
            if not txt:
                continue
            ch = _extract_channel(p)
            if ch == "analysis":
                analyses.append(txt)
            elif ch == "final":
                finals.append(txt)
            else:
                # parts without channel are usually user-facing; collect as finals fallback
                finals.append(txt)
        fin = "".join(finals) if finals else None
        ana = "".join(analyses) if analyses else None
        return fin, ana

    if isinstance(content, dict):
        ch = _extract_channel(content)
        txt = _extract_text(content)
        if not txt:
            return None, None
        if ch == "analysis":
            return None, txt
        # default treat as final if not labeled analysis
        return txt, None

    # string or None -> no rewrite from "parts"
    return None, None


def _extract_channel(part: Any) -> Optional[str]:
    if not isinstance(part, dict):
        return None
    ch = part.get("channel")
    # Some providers may use different keys or types; normalize to lower-case str
    return str(ch).lower() if isinstance(ch, str) else None


def _extract_text(part: Any) -> Optional[str]:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        # Most common keys:
        if isinstance(part.get("text"), str):
            return part["text"]
        if isinstance(part.get("content"), str):
            return part["content"]
        # Some variants wrap text like {"type":"text","text": "..."}
        if isinstance(part.get("delta"), str):  # rare delta variants
            return part["delta"]
    return None


def _extract_reasoning_field(reasoning: Any) -> Optional[str]:
    """Normalize provider-style 'reasoning' fields to a flat string."""
    if reasoning is None:
        return None
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, dict):
        # common forms: {"type":"reasoning_text","text":"..."} or {"text":"..."}
        return _extract_text(reasoning)
    if isinstance(reasoning, list):
        # join any text-like entries
        return _join_texts(reasoning)
    return None


def _join_texts(parts: List[Any]) -> str:
    out: List[str] = []
    for p in parts:
        t = _extract_text(p)
        if t:
            out.append(t)
    return "".join(out)


def _coalesce_texts(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return a + b
    return a or b
