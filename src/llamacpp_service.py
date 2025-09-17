from __future__ import annotations

import json
import os
import time
import subprocess
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Union

import httpx

from model_fetcher import ModelFetcher


def _quiet_httpx_logs() -> None:
    try:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    except Exception:
        pass


class LlamaCppService:
    """
    Launch llama.cpp (llama-server) and proxy OpenAI-compatible routes.

    Behavior:
      - Readiness gating with friendly countdown logs every HEALTH_LOG_INTERVAL_SEC.
      - Optional GPU snapshot lines via `nvidia-smi` while waiting.
      - Incoming requests block until the server is healthy (no premature 503s).
      - Extra stabilization: require consecutive /health 200s and a /v1/models check.
      - When LLAMA_DEBUG is truthy, log the full JSON request and JSON response.
      - FAST STREAMING: zero-copy SSE pass-through (no JSON parse/encode per chunk).
    """

    def __init__(self) -> None:
        _quiet_httpx_logs()

        # Ensure shards are present locally; choose first shard for -m
        first_shard = self._ensure_model_local_or_fail()
        self.model_path = str(first_shard)

        self.host = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
        self.port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self.base = f"http://{self.host}:{self.port}"

        # Timeouts / retries / logging cadence
        self.ready_timeout = self._env_int("LLAMA_READY_TIMEOUT_SEC", 1800)
        self.health_log_interval = self._env_int("HEALTH_LOG_INTERVAL_SEC", 10)
        # How long request handlers will wait for readiness before failing
        self.request_max_wait = self._env_int("OPENAI_MAX_WAIT_SEC", self.ready_timeout)
        # Print GPU snapshots during readiness wait
        self.gpu_snapshot = self._env_bool("GPU_SNAPSHOT", True)
        # conditional debug of request/response JSON
        self.debug = self._env_bool("LLAMA_DEBUG", False)

        # Start server and wait until ready
        self.proc: Optional[subprocess.Popen] = None
        self._start_llama_server()

        self.max_concurrency = self._env_int("RUNPOD_MAX_CONCURRENCY", 1)

    # ---------- OpenAI-compatible entry points ----------
    async def handle_openai_route(
        self, route: str, route_input: Optional[Dict[str, Any]]
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        route = (route or "").strip().lower()
        data = self._with_default_params(route_input or {})

        if route.endswith("/v1/models"):
            return {
                "object": "list",
                "data": [{"id": Path(self.model_path).name, "object": "model"}],
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

    # ---------- HTTP proxy with readiness gating & debug logging ----------
    async def _proxy_openai(
        self, path: str, payload: Dict[str, Any]
    ) -> Union[Dict[str, Any], AsyncGenerator[Union[str, Dict[str, Any]], None]]:
        """
        Do not forward the call until the server is healthy; callers won't see a 503
        during model load. Bounded by OPENAI_MAX_WAIT_SEC.
        When LLAMA_DEBUG is truthy, log the full JSON request and the JSON response.
        """
        await self._await_ready(self.request_max_wait)
        url = f"{self.base}{path}"
        stream = bool(payload.get("stream", False))

        # --- DEBUG: log outgoing payload ---
        if self.debug:
            try:
                print(f"[llama-debug] -> POST {url}\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
            except Exception:
                print(f"[llama-debug] -> POST {url} (payload JSON-encode failed)")

        if stream:
            async def _gen() -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
                headers = {"Accept": "text/event-stream"}
                async with httpx.AsyncClient(timeout=None) as client:
                    # first attempt
                    async with client.stream("POST", url, json=payload, headers=headers) as resp:
                        if resp.status_code == 503:
                            # wait again and retry once
                            await self._await_ready(self.request_max_wait)
                            async with client.stream("POST", url, json=payload, headers=headers) as resp2:
                                resp2.raise_for_status()
                                async for line in resp2.aiter_lines():
                                    if not line or not line.startswith("data: "):
                                        continue
                                    # pass-through exactly as SSE; aiter_lines strips newlines -> add \n\n
                                    if line == "data: [DONE]":
                                        if self.debug:
                                            print("[llama-debug] <- [DONE]")
                                        yield "data: [DONE]\n\n"
                                        break
                                    if self.debug:
                                        # try to show the JSON chunk, but don't parse for speed
                                        print(f"[llama-debug] <- chunk {line[6:][:200]}...")
                                    yield line + "\n\n"
                            return

                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            if line == "data: [DONE]":
                                if self.debug:
                                    print("[llama-debug] <- [DONE]")
                                yield "data: [DONE]\n\n"
                                break
                            if self.debug:
                                print(f"[llama-debug] <- chunk {line[6:][:200]}...")
                            yield line + "\n\n"
            return _gen()

        # Non-streaming path
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=payload)
            if r.status_code == 503:
                await self._await_ready(self.request_max_wait)
                r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

            # --- DEBUG: log response JSON ---
            if self.debug:
                try:
                    print(f"[llama-debug] <- {r.status_code} {url}\n{json.dumps(data, ensure_ascii=False, indent=2)}")
                except Exception:
                    print(f"[llama-debug] <- {r.status_code} {url} (response JSON-encode failed)")
            return data

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

        # Temperature (default 1.0)
        temp = os.getenv("LLAMA_ARG_TEMP")
        if temp:
            args += ["--temp", str(temp)]
        else:
            args += ["--temp", "1.0"]

        # Min-p (default 0.0)
        min_p = os.getenv("LLAMA_ARG_MIN_P")
        if min_p:
            args += ["--min-p", str(min_p)]
        else:
            args += ["--min-p", "0.0"]

        # Top-p (default 1.0)
        top_p = os.getenv("LLAMA_ARG_TOP_P")
        if top_p:
            args += ["--top-p", str(top_p)]
        else:
            args += ["--top-p", "1.0"]

        # Top-k (default 0)
        top_k = os.getenv("LLAMA_ARG_TOP_K")
        if top_k:
            args += ["--top-k", str(top_k)]
        else:
            args += ["--top-k", "0"]

        # Jinja templating
        jinja = os.getenv("LLAMA_ARG_JINJA")
        if jinja and jinja.lower() not in ("0", "false", "no", "n", ""):
            args += ["--jinja"]
        elif jinja is None:  # Default to enabled
            args += ["--jinja"]

        # Cache type K (single dash)
        ctk = os.getenv("LLAMA_ARG_CTK")
        if ctk:
            args += ["-ctk", str(ctk)]
        else:
            args += ["-ctk", "q4_0"]

        # Cache type V (single dash)
        ctv = os.getenv("LLAMA_ARG_CTV")
        if ctv:
            args += ["-ctv", str(ctv)]
        else:
            args += ["-ctv", "q4_0"]

        # Batch size for prompt processing
        ub = os.getenv("LLAMA_ARG_UB")
        if ub:
            args += ["-ub", str(ub)]
        else:
            args += ["-ub", "2048"]

        # Batch size
        batch = os.getenv("LLAMA_ARG_BATCH")
        if batch:
            args += ["-b", str(batch)]
        else:
            args += ["-b", "2048"]

        # Flash attention
        fa = os.getenv("LLAMA_ARG_FA")
        if fa and fa.lower() not in ("0", "false", "no", "n", ""):
            args += ["-fa"]
        elif fa is None:  # Default to enabled
            args += ["-fa"]

        # Context size (default 131072)
        n_ctx = os.getenv("LLAMA_ARG_N_CTX") or os.getenv("N_CTX")
        if n_ctx:
            args += ["--ctx-size", str(n_ctx)]
        else:
            args += ["--ctx-size", "131072"]

        # GPU layers (default 999)
        ngl = os.getenv("LLAMA_ARG_N_GPU_LAYERS") or os.getenv("N_GPU_LAYERS")
        if ngl:
            args += ["--n-gpu-layers", str(ngl)]
        else:
            args += ["--n-gpu-layers", "999"]

        # Tensor split (default 0.5,0.5)
        ts = os.getenv("LLAMA_ARG_TENSOR_SPLIT") or os.getenv("TENSOR_SPLIT")
        if ts:
            args += ["--tensor-split", ts]
        else:
            args += ["--tensor-split", "0.5,0.5"]

        # Split mode (default layer)
        sm = os.getenv("LLAMA_ARG_SPLIT_MODE")
        if sm:
            args += ["--split-mode", sm]
        else:
            args += ["--split-mode", "layer"]

        # Parallel requests (default 8)
        n_par = os.getenv("LLAMA_ARG_N_PARALLEL")
        if n_par:
            args += ["--parallel", str(n_par)]
        else:
            args += ["--parallel", "8"]

        # No webui
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

        # Wait for health to be truly ready with friendly countdown logging
        self._wait_until_ready(self.ready_timeout)

    def _wait_until_ready(self, timeout: int) -> None:
        start = time.time()
        next_log = 0.0
        consecutive_ok = 0
        required_ok = 3  # reduce 200->503 flapping

        while True:
            elapsed = time.time() - start
            remaining = int(max(0, timeout - elapsed))

            # Periodic friendly log
            if elapsed >= next_log:
                print(f"[startup] Waiting {remaining}s for {self.base}/health to become available...")
                if self.gpu_snapshot:
                    self._log_gpu_snapshot()
                next_log = elapsed + self.health_log_interval

            # Health probe (once per second)
            ok = self._health_ok()
            if ok:
                consecutive_ok += 1
                if consecutive_ok >= required_ok:
                    # extra check: /v1/models should succeed
                    if self._models_ok():
                        # final info line (once)
                        self._log_health_once()
                        return
            else:
                consecutive_ok = 0

            if remaining <= 0:
                raise RuntimeError(
                    f"llama-server failed to become ready within {timeout}s "
                    f"(last health status was not stable 200)."
                )

            time.sleep(1.0)

    def _health_ok(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base}/health")
                return r.status_code == 200
        except Exception:
            return False

    def _models_ok(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.base}/v1/models")
                return r.status_code == 200
        except Exception:
            return False

    async def _await_ready(self, max_wait: int) -> None:
        """
        Async variant used in request path. Blocks caller until healthy or timeout.
        """
        start = time.time()
        last_log = -999
        while True:
            if self._health_ok():
                # double check models endpoint once
                if self._models_ok():
                    return
            waited = int(time.time() - start)
            if waited >= max_wait:
                raise RuntimeError("Model is still loading (timed out waiting for readiness).")
            # print a soft notice approximately every 10s to avoid chatty logs
            if self.health_log_interval > 0 and waited // self.health_log_interval != last_log:
                remaining = int(max_wait - waited)
                print(f"[request] Waiting {remaining}s for {self.base}/health to become available...")
                last_log = waited // self.health_log_interval
            await self._asleep(1.0)

    @staticmethod
    async def _asleep(sec: float) -> None:
        import asyncio
        await asyncio.sleep(sec)

    def _log_health_once(self) -> None:
        try:
            r = httpx.get(f"{self.base}/health", timeout=5.0)
            txt = (r.text or "").strip()
            print(f"[llama-server] READY /health {r.status_code}: {txt[:500]}")
        except Exception as e:
            print(f"[llama-server] /health log failed: {e}")

    # ---------- GPU snapshot logging ----------
    def _log_gpu_snapshot(self) -> None:
        """
        Emit a one-line summary per GPU: index, name, util%, mem used/total (and %).
        Uses `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.
        """
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(" ".join(cmd), shell=True, text=True, stderr=subprocess.STDOUT)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            for line in lines:
                # Example (nounits): "0, NVIDIA A100-SXM4-80GB, 12, 4235, 81251"
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue
                idx, name, util, mem_used_mb, mem_total_mb = parts
                try:
                    util_i = int(util)
                    used = float(mem_used_mb) / 1024.0
                    total = float(mem_total_mb) / 1024.0
                    pct = (used / total * 100.0) if total > 0 else 0.0
                    print(f"[gpu] GPU{idx} {name}: util={util_i}% vram={used:.1f}/{total:.1f} GB ({pct:.0f}%)")
                except Exception:
                    print(f"[gpu] {line}")
        except Exception as e:
            print(f"[gpu] nvidia-smi unavailable or failed: {e}")

    # ---------- model discovery / fetching ----------
    def _ensure_model_local_or_fail(self) -> Path:
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
            default_max = self._env_int("DEFAULT_MAX_TOKENS", 32768)
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

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except Exception:
            return default

    @staticmethod
    def _env_bool(key: str, default: bool) -> bool:
        val = os.getenv(key, "")
        if not val:
            return default
        return val.strip().lower() in ("1", "true", "yes", "y")

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
