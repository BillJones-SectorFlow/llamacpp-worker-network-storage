from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore import UNSIGNED
from boto3.s3.transfer import TransferConfig


class ModelFetcher:
    """
    Ensures GGUF shard files exist locally (prefer /runpod-volume/models/<subdir>).
    If ALL required files are already present, returns immediately without any S3 calls.
    """

    def __init__(self) -> None:
        # Endpoint / auth (lazy client creation so we can early-return when cached)
        self.endpoint = os.getenv("B2_S3_ENDPOINT", "https://s3.us-west-004.backblazeb2.com").strip()
        self.key_id = (os.getenv("B2_KEY_ID") or "").strip()
        self.app_key = (os.getenv("B2_APP_KEY") or "").strip()

        # Where are the files
        self.bucket = (os.getenv("B2_BUCKET") or "").strip()
        self.prefix = (os.getenv("B2_PREFIX") or "").strip().lstrip("/")

        # Explicit file list OR generated from basename+shards
        self.model_files = self._parse_file_list(os.getenv("MODEL_FILES", ""))
        self.basename = (os.getenv("MODEL_BASENAME") or "").strip()
        self.shards = int(os.getenv("MODEL_SHARDS", "0") or "0")

        # Local cache root and subdir
        self.local_root = Path(os.getenv("MODEL_LOCAL_DIR", "") or self._default_local_root())
        subdir = (os.getenv("MODEL_SUBDIR") or (self.prefix.rsplit("/", 1)[-1] if self.prefix else "model")).strip()
        self.local_dir = self.local_root / subdir
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Transfer tuning
        max_conc = int(os.getenv("B2_MAX_CONCURRENCY", "16") or "16")
        chunk_mb = int(os.getenv("B2_CHUNK_MB", "64") or "64")
        self.tx_cfg = TransferConfig(
            multipart_threshold=chunk_mb * 1024 * 1024,
            multipart_chunksize=chunk_mb * 1024 * 1024,
            max_concurrency=max_conc,
            use_threads=True,
        )

        self._s3 = None  # lazy

    # ---------- public API ----------

    def ensure_local(self) -> Path:
        """
        Ensure all required GGUF shards exist locally; return path to FIRST shard.
        Fast path: if all files exist in /runpod-volume/models (or configured dir),
        skip any S3 activity entirely.
        """
        files = self._resolve_file_list()
        if not files:
            raise RuntimeError(
                "No model files resolved. Set MODEL_FILES or MODEL_BASENAME + MODEL_SHARDS, "
                "and B2_BUCKET/B2_PREFIX."
            )

        # Candidate directories to check for *already-present* shards:
        #  1) configured subdir (e.g., /runpod-volume/models/<subdir>)
        #  2) root cache dir (e.g., /runpod-volume/models) — this covers the
        #     case where files were placed directly under the root (your example).
        candidate_dirs = [self.local_dir, self.local_root]

        for cand in candidate_dirs:
            if self._all_locals_present(cand, files):
                first = self._first_shard_path(cand, files)
                # Re-point local_dir so llama-server can find sister shards in same folder
                self.local_dir = cand
                return first

        # If not fully present, download into self.local_dir
        for fname in files:
            key = "/".join(p for p in [self.prefix, fname] if p)
            local = self.local_dir / fname
            self._download_if_needed(key, local)

        return self._first_shard_path(self.local_dir, files)

    # ---------- helpers ----------

    def _resolve_file_list(self) -> List[str]:
        if self.model_files:
            return self.model_files

        if self.basename and self.shards > 0:
            width = max(5, len(str(self.shards)))  # typical pattern uses 5 digits
            return [
                f"{self.basename}-{str(i).zfill(width)}-of-{str(self.shards).zfill(width)}.gguf"
                for i in range(1, self.shards + 1)
            ]

        # As a convenience, list from prefix and take *.gguf if nothing explicit provided
        if self.bucket and self.prefix:
            ggufs = self._list_keys_with_suffix(".gguf")
            return [k.rsplit("/", 1)[-1] for k in ggufs]

        return []

    def _all_locals_present(self, base_dir: Path, files: List[str]) -> bool:
        """
        Return True if *every* file exists (>0 bytes) in base_dir.
        No remote checks; purely local to enable a zero-S3 fast path.
        """
        try:
            for fname in files:
                p = base_dir / fname
                if not (p.exists() and p.is_file() and p.stat().st_size > 0):
                    return False
            return True
        except Exception:
            return False

    def _first_shard_path(self, base_dir: Path, files: List[str]) -> Path:
        # Prefer the "-00001-of-XXXXX.gguf" shard if present; otherwise first lexicographically
        for fname in files:
            if self._is_first_shard(fname):
                return base_dir / fname
        return base_dir / sorted(files)[0]

    def _download_if_needed(self, key: str, local: Path) -> None:
        """
        Download object to 'local' if missing. If the file already exists locally,
        we TRUST it and return without any remote HEAD or re-download attempts.
        """
        if local.exists() and local.is_file() and local.stat().st_size > 0:
            return

        local.parent.mkdir(parents=True, exist_ok=True)

        backoff = 1.0
        for attempt in range(6):
            try:
                self._get_s3().download_file(
                    Bucket=self.bucket,
                    Key=key,
                    Filename=str(local),
                    Config=self.tx_cfg,
                )
                return
            except Exception as e:
                if attempt == 5:
                    raise RuntimeError(f"Download failed for s3://{self.bucket}/{key}: {e}") from e
                time.sleep(backoff)
                backoff *= 2.0

    def _list_keys_with_suffix(self, suffix: str) -> List[str]:
        keys: List[str] = []
        token: Optional[str] = None
        while True:
            resp = self._get_s3().list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix + ("" if self.prefix.endswith("/") else "/"),
                ContinuationToken=token,
            )
            for obj in resp.get("Contents", []) or []:
                k = obj["Key"]
                if k.lower().endswith(suffix):
                    keys.append(k)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return sorted(keys)

    def _get_s3(self):
        if self._s3 is not None:
            return self._s3
        if self.key_id and self.app_key:
            self._s3 = boto3.client(
                "s3",
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.app_key,
                endpoint_url=self.endpoint,
                region_name="us-west-004",
                config=BotoConfig(s3={"addressing_style": "virtual"}),
            )
        else:
            # Public buckets (unsigned)
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                region_name="us-west-004",
                config=BotoConfig(signature_version=UNSIGNED, s3={"addressing_style": "virtual"}),
            )
        return self._s3

    @staticmethod
    def _default_local_root() -> str:
        # Prefer RunPod network volume for cross-worker cache
        if Path("/runpod-volume").exists():
            return "/runpod-volume/models"
        return "/models"

    @staticmethod
    def _parse_file_list(csv: str) -> List[str]:
        items = [x.strip() for x in csv.split(",") if x.strip()]
        return [x for x in items if x.lower().endswith(".gguf")]

    @staticmethod
    def _is_first_shard(name: str) -> bool:
        # Match ...-00001-of-000NN.gguf
        return bool(re.search(r"-0*1-of-0*\d+\.gguf$", name, flags=re.IGNORECASE))
