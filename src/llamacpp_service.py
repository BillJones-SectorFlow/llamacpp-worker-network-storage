from __future__ import annotations

import glob
import json
import os
import time
import subprocess
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


class LlamaCppService:
    """
    Launches the native llama.cpp server (llama-server) with CUDA and proxies
    OpenAI-compatible routes to it.

    Key points:
      - Uses baked model if present under BASE_PATH (same pattern as RunPod worker-vllm).
      - Defaults max_tokens to 32768.
      - GPU offload via env passthrough: LLAMA_ARG_N_GPU_LAYERS, LLAMA_ARG_TENSOR_SPLIT, etc.
      - Logs a one-time /health payload when ready (no /v1/models check).
    """

    def __init__(self) -> None:
        # model path resolution may have been replaced by your B2/network-volume fetcher;
        # retain existing behavior if still present in your codebase.
        self.model_path = self._locate_baked_model_or_fail()
        self.host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self.base = f"http://{self.host}:{self.port}"

        self.proc: Optional[subprocess.Popen] = None
        self._start_llama_server()

        # stream formatting for RunPod:
        # - "text" (default): yield {"output": "<token>"} per chunk (best for RunPod SDKs /stream)
        # - "raw" : yield the original OpenAI chunk objects
        self.stream_format = os.getenv("LLAMA_STREAM_FORMAT", "text").strip().lower()
        if self.stream_format not in {"text", "raw"}:
            self.stream_format = "text"

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
            return {"object": "list",
                    "data": [{"id": os.path.basename(self.model_path), "object": "model"}]}

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
        debug = _env_truthy("LLAMA_DEBUG", "0")

        if stream:
            if debug:
                try:
                    print("[llama-debug] -> POST", url, flush=True)
                    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
                except Exception:
                    pass

            async def _gen() -> AsyncGenerator[Dict[str, Any], None]:
                # stream SSE from llama-server, translate to RunPod-friendly yields
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if not line.startswith("data: "):
                                continue
                            chunk = line[len("data: "):].strip()
                            if chunk == "[DONE]":
                                if debug:
                                    print("[llama-debug] <- [DONE]", flush=True)
                                break
                            try:
                                obj = json.loads(chunk)
                            except Exception:
                                continue

                            # Optional debug of raw chunk
                            if debug:
                                try:
                                    print("[llama-debug] <- chunk", flush=True)
                                    print(json.dumps(obj, indent=2, ensure_ascii=False), flush=True)
                                except Exception:
                                    pass

                            if self.stream_format == "raw":
                                # Yield the original llama-server/OpenAI chunk object
                                yield obj
                                continue

                            # Default: extract text piece and yield {"output": "<token>"}
                            text_piece = self._extract_text_piece(obj)
                            if text_piece:
                                yield {"output": text_piece}

                # generator ends → RunPod marks stream complete

            return _gen()

        # Non-streaming: one-shot POST, return JSON as-is
        async with httpx.AsyncClient(timeout=None) as client:
            if debug:
                try:
                    print("[llama-debug] -> POST", url, flush=True)
                    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
                except Exception:
                    pass
            r = await client.post(url, json=payload)
            r.raise_for_status()
            out = r.json()
            if debug:
                try:
                    print("[llama-debug] <- response", flush=True)
                    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
                except Exception:
                    pass
            return out

    @staticmethod
    def _extract_text_piece(chunk: Dict[str, Any]) -> str:
        """
        Pull token text from either chat (`choices[0].delta.content`) or
        legacy completions (`choices[0].text`). Return empty string if none.
        """
        try:
            choices = chunk.get("choices") or []
            if not choices:
                return ""
            c0 = choices[0] or {}
            delta = c0.get("delta") or {}
            if isinstance(delta, dict):
                txt = delta.get("content")
                if isinstance(txt, str):
                    return txt
            # fallback: text field (completions)
            txt2 = c0.get("text")
            if isinstance(txt2, str):
                return txt2
        except Exception:
            pass
        return ""

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
                    # one-time health log snippet
                    self._log_health_once()
                    return
            except Exception:
                pass
            time.sleep(0.8)

        # surface last lines of output if startup failed
        try:
            if self.proc and self.proc.stdout:
                lines = []
                for _ in range(120):
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    lines.append(line.rstrip())
                raise RuntimeError("llama-server failed to become ready.\n" + "\n".join(lines))
        except Exception as e:
            raise e

    def _log_health_once(self) -> None:
        """Print a short /health payload to stdout for easier diagnostics."""
        try:
            r = httpx.get(f"{self.base}/health", timeout=5.0)
            txt = (r.text or "").strip()
            # keep it concise in logs
            print(f"[llama-server] /health {r.status_code}: {txt[:500]}")
        except Exception as e:
            print(f"[llama-server] /health log failed: {e}")

    # ---------- model discovery ----------
    def _locate_baked_model_or_fail(self) -> str:
        """
        Prefer GGUF files baked into the image under BASE_PATH.
        If MODEL_VARIANT is set, prefer files whose path includes that variant.
        Pick the largest .gguf.
        """
        base = (os.getenv("BASE_PATH") or "/models").rstrip("/")
        variant = (os.getenv("MODEL_VARIANT") or "").strip()

        candidates = self._find_all_gguf(base)
        if not candidates:
            raise RuntimeError(
                f"No .gguf files found under BASE_PATH={base}. "
                f"Make sure the image was built with the model baked in."
            )

        if variant:
            v = [p for p in candidates if f"/{variant}/" in p or p.endswith(f"/{variant}.gguf")]
            if v:
                candidates = v

        candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return candidates[0]

    @staticmethod
    def _find_all_gguf(root: str) -> List[str]:
        pats = [os.path.join(root, "**", "*.gguf"), os.path.join(root, "*.gguf")]
        out: List[str] = []
        for p in pats:
            out.extend(glob.glob(p, recursive=True))
        seen: set[str] = set()
        uniq: List[str] = []
        for f in out:
            if os.path.isfile(f) and f not in seen:
                uniq.append(f)
                seen.add(f)
        return uniq

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
