from __future__ import annotations

import inspect
import os
import traceback
from typing import Any, AsyncGenerator, Dict, Optional

import runpod

from llamacpp_service import LlamaCppService
from utils import create_error_response

# Instantiate the service at import time (one llama-server per container)
try:
    llama_service = LlamaCppService()
except Exception as e:
    traceback.print_exc()
    err_msg = f"startup failed: {e}"

    class _BrokenService:
        """
        Minimal stub that surfaces the startup error on any call.
        Avoids illegal function signatures by storing err on self.
        """
        max_concurrency = 1

        def __init__(self, msg: str) -> None:
            self._err = msg

        async def handle_openai_route(self, *args, **kwargs):
            raise RuntimeError(self._err)

        async def generate(self, *args, **kwargs):
            raise RuntimeError(self._err)

    llama_service = _BrokenService(err_msg)  # type: ignore


async def handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    RunPod async-generator handler.

    Expected job payloads:

      - {"input": {"openai_route": "/v1/models"}}
      - {"input": {"openai_route": "/v1/chat/completions", "openai_input": {...}}}
      - {"input": {"prompt": "...", "stream": true, "temperature": 0.2, ...}}
      - {"input": {"messages": [...], "stream": false, ...}}
    """
    job_input: Dict[str, Any] = (job or {}).get("input", {}) or {}
    try:
        # Dispatch OpenAI-compatible routes first
        if "openai_route" in job_input:
            route = job_input.get("openai_route")
            openai_input = job_input.get("openai_input")
            result = await llama_service.handle_openai_route(route, openai_input)
            if inspect.isasyncgen(result):
                async for chunk in result:
                    yield chunk
            else:
                yield result
            return

        # Raw convenience API (prompt/messages)
        result = await llama_service.generate(job_input)
        if inspect.isasyncgen(result):
            async for chunk in result:
                yield chunk
        else:
            yield result

    except Exception as e:
        traceback.print_exc()
        yield create_error_response(e)


def _concurrency_modifier(_job: Optional[Dict[str, Any]] = None) -> int:
    # Allow env override but default to service-provided number
    try:
        env_override = int(os.getenv("RUNPOD_MAX_CONCURRENCY", "").strip() or "0")
    except Exception:
        env_override = 0
    return env_override if env_override > 0 else getattr(llama_service, "max_concurrency", 1)


# Start the serverless worker
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": _concurrency_modifier,
        "return_aggregate_stream": True,  # stream aggregation back to client
    }
)
