#!/usr/bin/env python3
"""
SURFACE Cluster Preflight Verification Script.

Read-only diagnostic utility to verify network reachability, Redis connectivity,
worker token authentication, Python dependencies, model checkpoint integrity,
and directory structure before launching cluster master or worker nodes.

Authoritative specification: SURFACE M4b Phase 4.1.

Usage:
  python3 scripts/check_cluster_preflight.py [--env-file PATH] [--role master|worker|auto]
Examples:
  python3 scripts/check_cluster_preflight.py --env-file backend/.env.master --role master
  python3 scripts/check_cluster_preflight.py --env-file backend/.env.worker --role worker
  python3 scripts/check_cluster_preflight.py --role auto
"""

import argparse
import importlib
import os
from pathlib import Path
import re
import socket
import sys
from typing import Dict, List, Optional, Tuple
import urllib.parse


class CheckStatus:
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class CheckResult:
    def __init__(self, category: str, name: str, status: str, message: str, details: Optional[str] = None):
        self.category = category
        self.name = name
        self.status = status
        self.message = message
        self.details = details

    def __repr__(self) -> str:
        return f"<CheckResult {self.category}::{self.name} [{self.status}]: {self.message}>"


def mask_url_secret(url_str: str) -> str:
    """Masks password in Redis or HTTP URLs for safe logging."""
    if not url_str:
        return ""
    try:
        parsed = urllib.parse.urlsplit(url_str)
        if parsed.password:
            netloc = f"{parsed.username or ''}:****@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        return url_str
    except Exception:
        return "[URL parsing failed]"


def parse_env_file(filepath: Path) -> Dict[str, str]:
    """Parses environment variables from an env file without polluting os.environ."""
    env_vars: Dict[str, str] = {}
    if not filepath.exists() or not filepath.is_file():
        return env_vars

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Remove surrounding single or double quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                env_vars[key] = val
    return env_vars


def check_python_packages(role: str) -> List[CheckResult]:
    """Verifies presence of required Python packages."""
    results: List[CheckResult] = []

    # Essential packages for both master and worker
    core_packages = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server engine"),
        ("httpx", "Synchronous/Asynchronous HTTP client"),
        ("pydantic", "Data contract validation"),
        ("numpy", "Numerical array processing"),
        ("scipy", "Scientific algorithms"),
        ("celery", "Distributed task queue"),
        ("redis", "Redis driver and client"),
    ]

    for pkg_name, desc in core_packages:
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            results.append(CheckResult("Dependencies", f"Package {pkg_name}", CheckStatus.PASS, f"{desc} installed (v{ver})"))
        except ImportError:
            status = CheckStatus.FAIL if (role == "master" or pkg_name in ("celery", "redis", "httpx")) else CheckStatus.WARN
            results.append(CheckResult("Dependencies", f"Package {pkg_name}", status, f"Missing required dependency: {desc}"))

    # Computer Vision & ML dependencies
    ml_packages = [
        ("cv2", "OpenCV image processing"),
        ("torch", "PyTorch deep learning runtime"),
        ("torchvision", "TorchVision models and transformations"),
        ("open3d", "Open3D point cloud & mesh algorithms"),
    ]

    for pkg_name, desc in ml_packages:
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            results.append(CheckResult("Dependencies", f"Package {pkg_name}", CheckStatus.PASS, f"{desc} installed (v{ver})"))
        except ImportError:
            # Workers doing inference strictly require cv2 and torch; open3d required for stages 0-3
            results.append(CheckResult("Dependencies", f"Package {pkg_name}", CheckStatus.WARN, f"Optional or worker dependency missing: {desc}"))

    return results


def check_filesystem_and_model(repo_root: Path, model_path_str: Optional[str]) -> List[CheckResult]:
    """Verifies critical repository directories and model checkpoint."""
    results: List[CheckResult] = []

    # Repository structure
    backend_dir = repo_root / "backend"
    storage_dir = backend_dir / "storage"
    app_dir = backend_dir / "app"

    if app_dir.exists() and app_dir.is_dir():
        results.append(CheckResult("Filesystem", "Backend app directory", CheckStatus.PASS, f"Found at {app_dir}"))
    else:
        results.append(CheckResult("Filesystem", "Backend app directory", CheckStatus.FAIL, f"Missing {app_dir}"))

    if storage_dir.exists() and storage_dir.is_dir():
        results.append(CheckResult("Filesystem", "Storage directory", CheckStatus.PASS, f"Found at {storage_dir}"))
    else:
        results.append(CheckResult("Filesystem", "Storage directory", CheckStatus.WARN, f"Storage dir {storage_dir} will be auto-created on startup"))

    # Model checkpoint validation
    candidates = []
    if model_path_str:
        p = Path(model_path_str)
        if not p.is_absolute():
            candidates.append(repo_root / p)
        else:
            candidates.append(p)
    candidates.extend([
        repo_root / "backend" / "model_store" / "best_model_sdnet2018.pth",
        repo_root / "backend" / "models_store" / "best_model_sdnet2018.pth",
    ])

    found_model: Optional[Path] = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            found_model = cand
            break

    if found_model:
        size_bytes = found_model.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)
        if size_mb > 5.0:
            results.append(CheckResult("Filesystem", "Model checkpoint", CheckStatus.PASS, f"Found checkpoint at {found_model.relative_to(repo_root)} ({size_mb} MB)"))
        else:
            results.append(CheckResult("Filesystem", "Model checkpoint", CheckStatus.WARN, f"Checkpoint at {found_model} appears unusually small ({size_mb} MB)"))
    else:
        results.append(CheckResult(
            "Filesystem",
            "Model checkpoint",
            CheckStatus.WARN,
            "best_model_sdnet2018.pth not found in backend/model_store. Crack classification will fall back to heuristic mode."
        ))

    return results


def check_configuration_values(env: Dict[str, str], role: str) -> List[CheckResult]:
    """Validates configuration parameters and token constraints."""
    results: List[CheckResult] = []

    exec_mode = env.get("SURFACE_EXECUTION", "local").strip().lower()
    if exec_mode == "celery":
        results.append(CheckResult("Configuration", "Execution mode", CheckStatus.PASS, "SURFACE_EXECUTION=celery (Distributed)"))
    else:
        results.append(CheckResult("Configuration", "Execution mode", CheckStatus.WARN if role == "worker" else CheckStatus.PASS, f"SURFACE_EXECUTION={exec_mode} (Thread Pool)"))

    # Worker token check
    token = env.get("SURFACE_WORKER_TOKEN", "").strip()
    if not token:
        if exec_mode == "celery":
            results.append(CheckResult("Security", "SURFACE_WORKER_TOKEN", CheckStatus.FAIL, "Token is empty! Master will reject all worker operations with HTTP 403."))
        else:
            results.append(CheckResult("Security", "SURFACE_WORKER_TOKEN", CheckStatus.WARN, "Token is unset (acceptable in local mode)."))
    elif len(token) < 8:
        results.append(CheckResult("Security", "SURFACE_WORKER_TOKEN", CheckStatus.WARN, f"Token configured but short ({len(token)} chars); recommend >= 16 chars."))
    else:
        results.append(CheckResult("Security", "SURFACE_WORKER_TOKEN", CheckStatus.PASS, f"Token configured securely (length: {len(token)} chars, secret hidden)."))

    # Worker Name check
    worker_name = env.get("SURFACE_WORKER_NAME", "").strip()
    if worker_name:
        results.append(CheckResult("Configuration", "SURFACE_WORKER_NAME", CheckStatus.PASS, f"Worker identity: '{worker_name}'"))
    else:
        results.append(CheckResult("Configuration", "SURFACE_WORKER_NAME", CheckStatus.PASS, f"Worker identity default: '{socket.gethostname()}'"))

    # CORS check on master
    if role in ("master", "auto"):
        cors = env.get("SURFACE_CORS_ORIGINS", "").strip()
        if cors:
            results.append(CheckResult("Configuration", "SURFACE_CORS_ORIGINS", CheckStatus.PASS, f"Configured extra origins: {cors}"))
        else:
            results.append(CheckResult("Configuration", "SURFACE_CORS_ORIGINS", CheckStatus.PASS, "Default CORS origins enabled (localhost:5173, localhost:3000)"))

    return results


def check_redis_connectivity(redis_url: str, timeout: float = 3.0) -> List[CheckResult]:
    """Tests TCP connection and Redis authentication against the broker URL."""
    results: List[CheckResult] = []
    masked = mask_url_secret(redis_url)

    try:
        parsed = urllib.parse.urlsplit(redis_url)
    except Exception as exc:
        results.append(CheckResult("Network", "Redis URL format", CheckStatus.FAIL, f"Invalid URL '{masked}': {exc}"))
        return results

    if parsed.scheme not in ("redis", "rediss"):
        results.append(CheckResult("Network", "Redis URL scheme", CheckStatus.FAIL, f"Unsupported scheme '{parsed.scheme}'; expected 'redis://' or 'rediss://'"))
        return results

    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379

    # 1. Low-level TCP socket test
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.close()
        results.append(CheckResult("Network", "Redis TCP connect", CheckStatus.PASS, f"Connected to {host}:{port} ({masked})"))
    except socket.timeout:
        s.close()
        results.append(CheckResult("Network", "Redis TCP connect", CheckStatus.FAIL, f"Timed out connecting to {host}:{port} after {timeout}s"))
        return results
    except ConnectionRefusedError:
        s.close()
        results.append(CheckResult("Network", "Redis TCP connect", CheckStatus.FAIL, f"Connection refused by {host}:{port}. Is Redis running and bound to this interface?"))
        return results
    except Exception as exc:
        s.close()
        results.append(CheckResult("Network", "Redis TCP connect", CheckStatus.FAIL, f"TCP connection error to {host}:{port}: {exc}"))
        return results

    # 2. Redis protocol PING if redis module is installed
    try:
        import redis as redis_lib
        client = redis_lib.from_url(redis_url, socket_connect_timeout=timeout, socket_timeout=timeout)
        try:
            pong = client.ping()
            if pong:
                results.append(CheckResult("Network", "Redis PING command", CheckStatus.PASS, "Redis broker responded with PONG"))
        except redis_lib.AuthenticationError as auth_err:
            results.append(CheckResult("Security", "Redis authentication", CheckStatus.FAIL, f"Redis authentication failed: {auth_err}"))
        except Exception as cmd_err:
            results.append(CheckResult("Network", "Redis PING command", CheckStatus.FAIL, f"Redis command failed: {cmd_err}"))
        finally:
            client.close()
    except ImportError:
        results.append(CheckResult("Network", "Redis client check", CheckStatus.WARN, "Python 'redis' package not installed; skipped protocol PING check"))

    return results


def check_master_api_health(master_url: str, timeout: float = 3.0) -> List[CheckResult]:
    """Tests Master API HTTP reachability and health endpoint."""
    results: List[CheckResult] = []
    clean_master = master_url.rstrip("/")

    try:
        import httpx
    except ImportError:
        results.append(CheckResult("Network", "HTTP client", CheckStatus.FAIL, "Python 'httpx' library not installed; cannot test HTTP endpoints"))
        return results

    health_url = f"{clean_master}/api/health"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(health_url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "healthy":
                    results.append(CheckResult("Network", "Master health check", CheckStatus.PASS, f"GET {health_url} -> 200 healthy"))
                else:
                    results.append(CheckResult("Network", "Master health check", CheckStatus.WARN, f"GET {health_url} -> 200 unexpected body: {data}"))
            else:
                results.append(CheckResult("Network", "Master health check", CheckStatus.FAIL, f"GET {health_url} returned HTTP {resp.status_code}"))
    except httpx.ConnectError:
        results.append(CheckResult("Network", "Master health check", CheckStatus.FAIL, f"Connection refused to {clean_master}. Is the FastAPI server running?"))
    except httpx.TimeoutException:
        results.append(CheckResult("Network", "Master health check", CheckStatus.FAIL, f"Connection to {clean_master} timed out after {timeout}s"))
    except Exception as exc:
        results.append(CheckResult("Network", "Master health check", CheckStatus.FAIL, f"Error querying {clean_master}: {exc}"))

    return results


def check_internal_api_authentication(master_url: str, worker_token: str, timeout: float = 3.0) -> List[CheckResult]:
    """
    Safely probes Master internal route authentication without mutating state.
    Uses GET /api/internal/jobs/probe-check/manifest.
    """
    results: List[CheckResult] = []
    clean_master = master_url.rstrip("/")

    try:
        import httpx
    except ImportError:
        return results

    probe_url = f"{clean_master}/api/internal/jobs/preflight-probe-nonexistent/manifest"

    # Probe 1: Without token -> Expected HTTP 403 Forbidden
    try:
        with httpx.Client(timeout=timeout) as client:
            resp_no_token = client.get(probe_url)
            if resp_no_token.status_code == 403:
                results.append(CheckResult("Security", "Internal API enforcement", CheckStatus.PASS, "Master rejected unauthenticated probe with HTTP 403 (Token enforced)"))
            else:
                results.append(CheckResult("Security", "Internal API enforcement", CheckStatus.FAIL, f"Master responded with HTTP {resp_no_token.status_code} instead of 403 Forbidden!"))
    except Exception as exc:
        results.append(CheckResult("Security", "Internal API enforcement", CheckStatus.FAIL, f"Failed unauthenticated security probe: {exc}"))
        return results

    # Probe 2: With configured token -> Expected token acceptance (409 Conflict for missing fencing headers or 404 for missing job)
    if not worker_token:
        results.append(CheckResult("Security", "Worker token verification", CheckStatus.FAIL, "No token supplied; cannot verify token match with master"))
        return results

    try:
        headers = {"X-Worker-Token": worker_token}
        with httpx.Client(timeout=timeout) as client:
            resp_with_token = client.get(probe_url, headers=headers)
            if resp_with_token.status_code == 403:
                results.append(CheckResult("Security", "Worker token verification", CheckStatus.FAIL, "Master rejected supplied token with HTTP 403 Forbidden (Token mismatch!)"))
            elif resp_with_token.status_code in (404, 409):
                results.append(CheckResult("Security", "Worker token verification", CheckStatus.PASS, f"Master accepted X-Worker-Token (HTTP {resp_with_token.status_code}: Token matched)"))
            else:
                results.append(CheckResult("Security", "Worker token verification", CheckStatus.WARN, f"Master returned unexpected HTTP {resp_with_token.status_code}"))
    except Exception as exc:
        results.append(CheckResult("Security", "Worker token verification", CheckStatus.FAIL, f"Failed authenticated security probe: {exc}"))

    return results


def run_preflight_checks(
    env_file: Optional[Path] = None,
    role: str = "auto",
    master_url_override: Optional[str] = None,
    redis_url_override: Optional[str] = None,
    worker_token_override: Optional[str] = None,
    timeout: float = 3.0,
    repo_root: Optional[Path] = None,
) -> Tuple[List[CheckResult], int, int, int]:
    """Runs all preflight verification checks and returns aggregated results."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    # Determine environment variables
    merged_env = dict(os.environ)
    if env_file and env_file.exists():
        parsed = parse_env_file(env_file)
        merged_env.update(parsed)

    if master_url_override:
        merged_env["SURFACE_MASTER_URL"] = master_url_override
    if redis_url_override:
        merged_env["REDIS_URL"] = redis_url_override
    if worker_token_override:
        merged_env["SURFACE_WORKER_TOKEN"] = worker_token_override

    # Auto-detect role
    if role == "auto":
        if "SURFACE_CORS_ORIGINS" in merged_env or "master" in str(env_file or "").lower():
            role = "master"
        elif "worker" in str(env_file or "").lower():
            role = "worker"
        else:
            role = "master"

    results: List[CheckResult] = []

    # 1. Python dependencies
    results.extend(check_python_packages(role))

    # 2. Filesystem & Model
    model_path = merged_env.get("SURFACE_MODEL_PATH")
    results.extend(check_filesystem_and_model(repo_root, model_path))

    # 3. Environment configuration & Security
    results.extend(check_configuration_values(merged_env, role))

    # 4. Redis Connectivity
    redis_url = merged_env.get("REDIS_URL", "redis://localhost:6379/0").strip()
    results.extend(check_redis_connectivity(redis_url, timeout=timeout))

    # 5. Master API & Authentication Probes (if in celery mode or checking worker role)
    master_url = merged_env.get("SURFACE_MASTER_URL", "http://localhost:8000").strip()
    token = merged_env.get("SURFACE_WORKER_TOKEN", "").strip()
    exec_mode = merged_env.get("SURFACE_EXECUTION", "local").strip().lower()

    if exec_mode == "celery" or role == "worker":
        results.extend(check_master_api_health(master_url, timeout=timeout))
        # Only probe internal auth if health succeeded
        health_passed = any(r.name == "Master health check" and r.status == CheckStatus.PASS for r in results)
        if health_passed:
            results.extend(check_internal_api_authentication(master_url, token, timeout=timeout))
        else:
            results.append(CheckResult("Security", "Worker token verification", CheckStatus.WARN, "Skipped internal token probe because master health check was unreachable"))

    pass_count = sum(1 for r in results if r.status == CheckStatus.PASS)
    warn_count = sum(1 for r in results if r.status == CheckStatus.WARN)
    fail_count = sum(1 for r in results if r.status == CheckStatus.FAIL)

    return results, pass_count, warn_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="SURFACE Cluster Preflight Verification Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Path to environment file (e.g. backend/.env.master)")
    parser.add_argument("--role", choices=["master", "worker", "auto"], default="auto", help="Cluster node role being verified")
    parser.add_argument("--master-url", type=str, default=None, help="Override master API URL")
    parser.add_argument("--redis-url", type=str, default=None, help="Override Redis broker URL")
    parser.add_argument("--worker-token", type=str, default=None, help="Override worker token")
    parser.add_argument("--timeout", type=float, default=3.0, help="Network probe timeout in seconds")
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output")

    args = parser.parse_args()

    # ANSI Colors
    use_color = sys.stdout.isatty() and not args.no_color
    GREEN = "\033[92m" if use_color else ""
    YELLOW = "\033[93m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    print(f"\n{BOLD}======================================================================{RESET}")
    print(f"{BOLD} SURFACE — Cluster Preflight Diagnostic Verification{RESET}")
    print(f"{BOLD}======================================================================{RESET}")
    if args.env_file:
        print(f" Loaded Env File: {args.env_file}")
    print(f" Target Role    : {args.role}")
    print(f" Probe Timeout  : {args.timeout}s")
    print(f"----------------------------------------------------------------------")

    results, pass_count, warn_count, fail_count = run_preflight_checks(
        env_file=args.env_file,
        role=args.role,
        master_url_override=args.master_url,
        redis_url_override=args.redis_url,
        worker_token_override=args.worker_token,
        timeout=args.timeout,
    )

    current_cat = ""
    for r in results:
        if r.category != current_cat:
            current_cat = r.category
            print(f"\n{BOLD}[{current_cat}]{RESET}")

        if r.status == CheckStatus.PASS:
            tag = f"{GREEN}[PASS]{RESET}"
        elif r.status == CheckStatus.WARN:
            tag = f"{YELLOW}[WARN]{RESET}"
        else:
            tag = f"{RED}[FAIL]{RESET}"

        print(f"  {tag} {r.name:<28} : {r.message}")
        if r.details:
            print(f"         {r.details}")

    print(f"\n{BOLD}----------------------------------------------------------------------{RESET}")
    summary_color = GREEN if fail_count == 0 else RED
    print(
        f" Preflight Summary: {summary_color}{BOLD}{pass_count} passed{RESET}, "
        f"{YELLOW}{warn_count} warnings{RESET}, "
        f"{RED if fail_count > 0 else ''}{fail_count} failures{RESET}"
    )
    print(f"{BOLD}======================================================================{RESET}\n")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
