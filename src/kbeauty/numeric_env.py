"""The pinned numeric environment of the canonical Linux run, and a record of the machine that produced a result.

GitHub's ubuntu-24.04 runners land on different CPUs. With AVX-512 available, numpy uses AVX-512 code paths
for exp/log/power and OpenBLAS picks its SkylakeX kernels; on other CPUs it uses AVX2/Haswell ones. The last
bits differ, and ETS, SARIMAX, ridge and the MLP amplify them into different forecasts
(docs/experiments/linux_determinism.md). PINS fixes the common denominator every runner supports (AVX2/FMA):
one BLAS thread, the Haswell OpenBLAS kernel and numpy without AVX-512 dispatch. The variables must be set
before numpy is imported, so they belong in the CI workflow's env, not in Python code.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PINS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_CORETYPE": "Haswell",
    "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR",
}
AVX512_DISPATCH = ("X86_V4", "AVX512_ICL", "AVX512_SPR")
PACKAGES = ["numpy", "scipy", "scikit-learn", "statsmodels", "pandas", "lightgbm", "xgboost", "joblib",
            "threadpoolctl", "duckdb", "dbt-core", "dbt-duckdb", "scipy-openblas64", "scipy-openblas32"]


def _cpu() -> dict:
    info = Path("/proc/cpuinfo")
    first = info.read_text().split("\n\n")[0] if info.exists() else ""
    field = {k.strip(): v.strip() for k, _, v in (ln.partition(":") for ln in first.splitlines())}
    flags = field.get("flags", "").split()
    return {
        "model": field.get("model name") or platform.processor(),
        "vendor": field.get("vendor_id"),
        "avx2": "avx2" in flags if flags else None,
        "fma": "fma" in flags if flags else None,
        "avx512_exposed": sorted(f for f in flags if f.startswith("avx512")),
        "flags_sha256_16": hashlib.sha256(" ".join(sorted(flags)).encode()).hexdigest()[:16] if flags else None,
        "logical_cpus": os.cpu_count(),
    }


def _git_commit(repo: Path) -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def record(repo: Path) -> dict:
    """Everything that decides the bits of a result: code, packages, BLAS, numpy dispatch, env, machine."""
    import numpy as np  # noqa: F401  (loads the BLAS so threadpoolctl can see it)
    import scipy.linalg  # noqa: F401
    from numpy._core import _multiarray_umath as mu
    from threadpoolctl import threadpool_info

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    server = os.environ.get("GITHUB_SERVER_URL")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return {
        "commit": _git_commit(repo),
        "workflow_run": (f"{server}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{run_id}"
                         if server and run_id else None),
        "runner": {
            "name": os.environ.get("RUNNER_NAME"),
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "platform": platform.platform(),
            "cpu": _cpu(),
        },
        "python": sys.version.split()[0],
        "packages": versions,
        "blas": [{k: d.get(k) for k in ("internal_api", "prefix", "version", "architecture", "num_threads",
                                         "threading_layer")} for d in threadpool_info()],
        "numpy_simd": {
            "baseline": list(mu.__cpu_baseline__),
            "dispatch": list(mu.__cpu_dispatch__),
            "enabled": sorted(k for k, v in mu.__cpu_features__.items() if v and k in mu.__cpu_dispatch__),
        },
        "env": {k: os.environ.get(k) for k in PINS},
    }


def pin_problems(rec: dict) -> list[str]:
    """Why a record does not come from the pinned environment (empty list = pinned)."""
    out = [f"{k}={rec['env'].get(k)!r} (expected {v!r})" for k, v in PINS.items() if rec["env"].get(k) != v]
    for lib in rec["blas"]:
        if lib.get("internal_api") == "openblas":
            if lib.get("num_threads") != 1:
                out.append(f"{lib['prefix']}: {lib['num_threads']} threads (expected 1)")
            if str(lib.get("architecture")).lower() != "haswell":
                out.append(f"{lib['prefix']}: kernel {lib.get('architecture')} (expected Haswell)")
    on = [f for f in AVX512_DISPATCH if f in rec["numpy_simd"]["enabled"]]
    if on:
        out.append(f"numpy AVX-512 dispatch enabled: {on}")
    return out


def run_label(rec: dict | None = None) -> str:
    """Which environment produced this run's results, for labels in tables, figures and documents. There is no
    fallback between environments: the official values are those of the pinned Linux environment only."""
    system = platform.system()
    if system == "Linux":
        rec = rec or record(Path(__file__).resolve().parents[2])
        return "Linux 고정 환경(공식)" if not pin_problems(rec) else "Linux 비고정 환경(비공식)"
    if system == "Windows":
        return "작성 환경(Windows, 비공식)"
    return f"{system} 환경(비공식)"
