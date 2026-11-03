"""
Unit and integration tests for SURFACE Cluster Preflight & Deployment Configuration.

Verifies:
- Detection of missing mandatory configuration in Celery mode.
- Strict secrecy (secrets are never leaked in messages or logs).
- Clean error reporting for unreachable master and failed Redis connections.
- Model checkpoint detection and heuristic fallback warnings.
- Mocked end-to-end PASS behavior for healthy master and worker nodes.
- Preserved backward compatibility for local execution mode.
- Shell script syntax and environment template integrity.
"""

import os
from pathlib import Path
import subprocess
from unittest.mock import patch, MagicMock
import pytest

from scripts.check_cluster_preflight import (
    run_preflight_checks,
    parse_env_file,
    mask_url_secret,
    CheckStatus,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def test_mask_url_secret_masks_redis_password():
    raw_url = "redis://:super_secret_password_123@192.168.56.10:6379/0"
    masked = mask_url_secret(raw_url)
    assert "super_secret_password_123" not in masked
    assert "****" in masked
    assert "192.168.56.10:6379/0" in masked


def test_mask_url_secret_handles_no_password():
    raw_url = "redis://192.168.56.10:6379/0"
    masked = mask_url_secret(raw_url)
    assert masked == raw_url


def test_preflight_missing_worker_token_in_celery_mode_fails(repo_root):
    env_overrides = {
        "SURFACE_EXECUTION": "celery",
        "SURFACE_WORKER_TOKEN": "",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
        "SURFACE_MASTER_URL": "http://127.0.0.1:8000",
    }
    with patch.dict(os.environ, env_overrides, clear=True):
        results, _, _, fail_count = run_preflight_checks(
            role="master",
            timeout=0.1,
            repo_root=repo_root,
        )
        token_results = [r for r in results if r.name == "SURFACE_WORKER_TOKEN"]
        assert len(token_results) == 1
        assert token_results[0].status == CheckStatus.FAIL
        assert "empty" in token_results[0].message.lower()


def test_preflight_never_prints_secrets(repo_root):
    secret_token = "TOP_SECRET_WORKER_TOKEN_ABCD1234EFGH"
    secret_pw = "TOP_SECRET_REDIS_PASSWORD_998877"
    redis_url = f"redis://:{secret_pw}@127.0.0.1:6379/0"

    env_overrides = {
        "SURFACE_EXECUTION": "celery",
        "SURFACE_WORKER_TOKEN": secret_token,
        "REDIS_URL": redis_url,
        "SURFACE_MASTER_URL": "http://127.0.0.1:8000",
    }
    with patch.dict(os.environ, env_overrides, clear=True):
        results, _, _, _ = run_preflight_checks(
            role="worker",
            timeout=0.1,
            repo_root=repo_root,
        )
        for r in results:
            full_text = f"{r.name} {r.message} {r.details or ''}"
            assert secret_token not in full_text, f"Secret token leaked in check '{r.name}': {full_text}"
            assert secret_pw not in full_text, f"Redis password leaked in check '{r.name}': {full_text}"


def test_preflight_unreachable_master_url_reported_as_failure(repo_root):
    env_overrides = {
        "SURFACE_EXECUTION": "celery",
        "SURFACE_WORKER_TOKEN": "valid_secret_token_12345",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
        "SURFACE_MASTER_URL": "http://127.0.0.1:59998",  # Non-existent master port
    }
    with patch.dict(os.environ, env_overrides, clear=True):
        results, _, _, _ = run_preflight_checks(
            role="worker",
            timeout=0.2,
            repo_root=repo_root,
        )
        master_checks = [r for r in results if r.name == "Master health check"]
        assert len(master_checks) == 1
        assert master_checks[0].status == CheckStatus.FAIL
        assert "connection refused" in master_checks[0].message.lower() or "error" in master_checks[0].message.lower()


def test_preflight_redis_connection_failure_reported_clearly(repo_root):
    env_overrides = {
        "SURFACE_EXECUTION": "celery",
        "SURFACE_WORKER_TOKEN": "valid_secret_token_12345",
        "REDIS_URL": "redis://127.0.0.1:59997/0",  # Non-existent Redis port
        "SURFACE_MASTER_URL": "http://127.0.0.1:8000",
    }
    with patch.dict(os.environ, env_overrides, clear=True):
        results, _, _, _ = run_preflight_checks(
            role="master",
            timeout=0.2,
            repo_root=repo_root,
        )
        redis_tcp = [r for r in results if r.name == "Redis TCP connect"]
        assert len(redis_tcp) == 1
        assert redis_tcp[0].status == CheckStatus.FAIL


def test_preflight_missing_model_checkpoint_reported_as_warning(repo_root, tmp_path):
    fake_empty_repo = tmp_path / "fake_repo"
    fake_empty_repo.mkdir()
    (fake_empty_repo / "backend" / "app").mkdir(parents=True)

    results, _, warn_count, _ = run_preflight_checks(
        role="worker",
        timeout=0.1,
        repo_root=fake_empty_repo,
    )
    model_checks = [r for r in results if r.name == "Model checkpoint"]
    assert len(model_checks) == 1
    assert model_checks[0].status == CheckStatus.WARN
    assert "not found" in model_checks[0].message.lower()


def test_preflight_local_mode_does_not_fail_on_missing_worker_token(repo_root):
    env_overrides = {
        "SURFACE_EXECUTION": "local",
        "SURFACE_WORKER_TOKEN": "",
        "REDIS_URL": "redis://127.0.0.1:6379/0",
    }
    with patch.dict(os.environ, env_overrides, clear=True):
        results, _, _, _ = run_preflight_checks(
            role="master",
            timeout=0.1,
            repo_root=repo_root,
        )
        token_results = [r for r in results if r.name == "SURFACE_WORKER_TOKEN"]
        assert len(token_results) == 1
        assert token_results[0].status != CheckStatus.FAIL


def test_preflight_successful_configuration_with_mocks(repo_root):
    env_overrides = {
        "SURFACE_EXECUTION": "celery",
        "SURFACE_WORKER_TOKEN": "production_quality_secret_token_2026",
        "SURFACE_WORKER_NAME": "worker-unit-test",
        "REDIS_URL": "redis://:secret_pass@127.0.0.1:6379/0",
        "SURFACE_MASTER_URL": "http://127.0.0.1:8000",
    }

    # Mock socket connect, redis client, and httpx client
    with patch.dict(os.environ, env_overrides, clear=True):
        with patch("socket.socket") as mock_sock_cls, \
             patch("httpx.Client") as mock_httpx_cls:

            # Mock socket connect success
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock

            # Mock httpx responses
            mock_client = MagicMock()
            mock_httpx_cls.return_value.__enter__.return_value = mock_client

            def mock_get(url, headers=None):
                mock_resp = MagicMock()
                if "/api/health" in url:
                    mock_resp.status_code = 200
                    mock_resp.json.return_value = {"status": "healthy"}
                elif "/manifest" in url:
                    if headers and headers.get("X-Worker-Token") == "production_quality_secret_token_2026":
                        mock_resp.status_code = 409  # Token matched, rejected on missing lease
                    else:
                        mock_resp.status_code = 403  # Missing/invalid token
                return mock_resp

            mock_client.get.side_effect = mock_get

            results, pass_count, warn_count, fail_count = run_preflight_checks(
                role="worker",
                timeout=1.0,
                repo_root=repo_root,
            )

            # Assert network & security checks pass
            net_and_sec = [r for r in results if r.category in ("Network", "Security")]
            for r in net_and_sec:
                if r.name in ("Master health check", "Internal API enforcement", "Worker token verification", "SURFACE_WORKER_TOKEN"):
                    assert r.status == CheckStatus.PASS, f"{r.name} failed: {r.message}"


def test_start_scripts_syntax_and_shebang(repo_root):
    scripts = [
        repo_root / "scripts" / "start_master.sh",
        repo_root / "scripts" / "start_worker.sh",
        repo_root / "scripts" / "check_cluster_preflight.py",
    ]

    for s in scripts:
        assert s.exists(), f"Script {s} does not exist"
        assert os.access(s, os.X_OK), f"Script {s} is not executable"
        content = s.read_text()
        assert content.startswith("#!/usr/bin/env"), f"Script {s} missing standard shebang"

        if s.suffix == ".sh":
            # Run bash -n syntax check
            res = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True)
            assert res.returncode == 0, f"Bash syntax check failed for {s}: {res.stderr}"


def test_env_example_templates_parse_cleanly(repo_root):
    master_example = repo_root / "backend" / ".env.master.example"
    worker_example = repo_root / "backend" / ".env.worker.example"

    assert master_example.exists(), "backend/.env.master.example is missing"
    assert worker_example.exists(), "backend/.env.worker.example is missing"

    master_env = parse_env_file(master_example)
    worker_env = parse_env_file(worker_example)

    # Check required keys in master template
    assert master_env.get("SURFACE_EXECUTION") == "celery"
    assert "REDIS_URL" in master_env
    assert "SURFACE_MASTER_URL" in master_env
    assert "SURFACE_WORKER_TOKEN" in master_env
    assert "SURFACE_WORKER_NAME" in master_env
    assert "SURFACE_CORS_ORIGINS" in master_env
    assert master_env.get("SURFACE_MODEL_PATH") == "backend/model_store/best_model_sdnet2018.pth"

    # Check required keys in worker template
    assert worker_env.get("SURFACE_EXECUTION") == "celery"
    assert "REDIS_URL" in worker_env
    assert "SURFACE_MASTER_URL" in worker_env
    assert "SURFACE_WORKER_TOKEN" in worker_env
    assert "SURFACE_WORKER_NAME" in worker_env
    assert worker_env.get("SURFACE_MODEL_PATH") == "backend/model_store/best_model_sdnet2018.pth"

    # Verify no real secrets were committed
    for env in (master_env, worker_env):
        token = env.get("SURFACE_WORKER_TOKEN", "")
        assert "CHANGE_ME" in token or "secret" in token.lower()
        redis_url = env.get("REDIS_URL", "")
        assert "CHANGE_ME" in redis_url or "password" in redis_url.lower()
