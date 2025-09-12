from __future__ import annotations

import glob
import json
import os
import time
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

from model_fetcher import ModelFetcher


class LlamaCppService:
    """
    Launches llama.cpp server (llama-server) with CUDA and proxies OpenAI routes.

    Changes vs. previous version:
      - Instead of assuming a baked-in model, we fetch GGUF shards from Backblaze B2
        at runtime (to /runpod-volume/models/<subdir> when available) and point
        llama-server to the FIRST shard (llama.cpp loads the remaining shards
        automatically when they are in the same directory).
    """

    def __init__(self) -> None:
        # Ensure model exists locally (download if needed)
        first_shard = self._ensure_model_local_or_fail()
        self.model_path = str(first_shard)

        self.host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self.base = f"http://{self.host}:{self.port}"

        self.proc: Optional[subprocess.Popen] = None
        self._start_llama_server()

        try:
            self.max_concurrency = int(os.getenv("RUNPOD_MAX_CONCURRENCY", "1"))
        except Exception:
            self.max_concurrency = 1

    # ---------- OpenAI-compatible entry points ----------
    async def handle_openai_route(
        self, route: str, route_input: Optional[Dict[str, Any]]
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        route = (route or "").strip().lower()
        data = self._with_default_params(route_input or {})

        if route.endswith("/v1/models"):
            return {
                "object": "list",
                "data": [
                    {"id": Path(self.model_path).name, "object": "model"}
                ],
            }

        if route.endswith("/v1/chat/completions"):
            return await self._proxy_openai("/v1/chat/completions", data)

        if route.endswith("/v1/completions"):
            return await self._proxy_openai("/v1/completions", data)

        return {"error": {"message": f"Unsupported route '{route}'.", "type": "route"}}

    async def generate(
        self, job_input: Dict[str, Any]
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        payload = self._with_default_params(job_input or {})
        if "messages" in payload:
            return await self._proxy_openai("/v1/chat/completions", payload)
        return await self._proxy_openai("/v1/completions", payload)

    # ---------- HTTP proxy ----------
    async def _proxy_openai(
        self, path: str, payload: Dict[str, Any]
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        url = f"{self.base}{path}"
        stream = bool(payload.get("stream", False))

        if stream:
            async def _gen() -> AsyncGenerator[Dict[str, Any], None]:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data: "):
                                chunk = line[len("data: "):].strip()
                                if chunk == "[DONE]":
                                    break
                                try:
                                    yield json.loads(chunk)
                                except Exception:
                                    pass
            return _gen()

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    # ---------- llama-server process mgmt ----------
    def _start_llama_server(self) -> None:
        if self.proc and self.proc.poll() is None:
            return

        args = [
            "llama-server",
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
        ]

        ngl = os.getenv("LLAMA_ARG_N_GPU_LAYERS") or os.getenv("N_GPU_LAYERS")
        if ngl:
            args += ["--n-gpu-layers", str(ngl)]

        ts = os.getenv("LLAMA_ARG_TENSOR_SPLIT") or os.getenv("TENSOR_SPLIT")
        if ts:
            args += ["--tensor-split", ts]

        sm = os.getenv("LLAMA_ARG_SPLIT_MODE")
        if sm:
            args += ["--split-mode", sm]

        n_ctx = os.getenv("LLAMA_ARG_N_CTX") or os.getenv("N_CTX")
        if n_ctx:
            args += ["--ctx-size", str(n_ctx)]

        if os.getenv("LLAMA_NO_WEBUI", "1") != "0":
            args += ["--no-webui"]

        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 120.0) -> None:
        """Wait only on /health, then log a one-time health snippet."""
        start = time.time()
        url_health = f"{self.base}/health"

        while time.time() - start < timeout:
            try:
                r = httpx.get(url_health, timeout=3.0)
                if r.status_code == 200:
                    self._log_health_once()
                    return
            except Exception:
                pass
            time.sleep(0.8)

        # surface output if startup failed
        try:
            if self.proc and self.proc.stdout:
                lines = []
                for _ in range(200):
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    lines.append(line.rstrip())
                raise RuntimeError("llama-server failed to become ready.\n" + "\n".join(lines))
        except Exception as e:
            raise e

    def _log_health_once(self) -> None:
        try:
            r = httpx.get(f"{self.base}/health", timeout=5.0)
            txt = (r.text or "").strip()
            print(f"[llama-server] /health {r.status_code}: {txt[:500]}")
        except Exception as e:
            print(f"[llama-server] /health log failed: {e}")

    # ---------- model discovery / fetching ----------
    def _ensure_model_local_or_fail(self) -> Path:
        """
        Prefer explicitly provided local path (MODEL_LOCAL_PATH). Otherwise fetch from B2.
        Return the path to the FIRST shard to pass to -m.
        """
        explicit = (os.getenv("MODEL_LOCAL_PATH") or "").strip()
        if explicit:
            p = Path(explicit)
            if not p.exists():
                raise RuntimeError(f"MODEL_LOCAL_PATH does not exist: {explicit}")
            return p

        fetcher = ModelFetcher()
        first = fetcher.ensure_local()
        if not first.exists():
            raise RuntimeError(f"Downloaded model shard not found: {first}")
        return first

    # ---------- defaults mapper ----------
    def _with_default_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(payload or {})
        if data.get("max_tokens") is None:
            try:
                default_max = int(os.getenv("DEFAULT_MAX_TOKENS", "32768"))
            except Exception:
                default_max = 32768
            data["max_tokens"] = max(1, default_max)

        # Normalize OpenAI "content parts" to plain text
        if "messages" in data and isinstance(data["messages"], list):
            for m in data["messages"]:
                c = m.get("content")
                if isinstance(c, list):
                    texts = []
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = part.get("text")
                            if isinstance(t, str):
                                texts.append(t)
                    m["content"] = "\n".join(texts)
        return data

    def __del__(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5.0)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass
