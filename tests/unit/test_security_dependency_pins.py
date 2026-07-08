# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails for NSPECT-UV6I-R3V9/NSPECT-S62Q-PZUD dependency remediation (pip-audit verified pins)."""

import pytest
from importlib.metadata import version
from packaging.version import Version


def test_cryptography_not_vulnerable_cve_2026_34073() -> None:
    # GHSA-537c-gmf6-5ccf: OpenSSL vuln in cryptography wheels
    assert Version(version("cryptography")) >= Version("48.0.1")


def test_pillow_not_vulnerable_cve_2026_42311() -> None:
    assert Version(version("pillow")) >= Version("12.2.0")


def test_urllib3_not_vulnerable_cve_2026_44432() -> None:
    assert Version(version("urllib3")) >= Version("2.7.0")


def test_transformers_not_vulnerable_cve_2026_1839() -> None:
    assert Version(version("transformers")) >= Version("5.0.0rc3")


def test_python_multipart_not_vulnerable_cve_2026_53538() -> None:
    # CVE-2026-53538/39/40: multiple multipart vulns; fix >= 0.0.31
    assert Version(version("python-multipart")) >= Version("0.0.31")


def test_orjson_not_vulnerable_ghsa_hx9q_6w63_j58v() -> None:
    assert Version(version("orjson")) >= Version("3.11.6")


def test_langsmith_not_vulnerable_ghsa_f4xh_w4cj_qxq8() -> None:
    # GHSA-f4xh-w4cj-qxq8: arbitrary file read via TracingMiddleware
    assert Version(version("langsmith")) >= Version("0.8.18")


def test_aiohttp_not_vulnerable_cve_2026_50269() -> None:
    # CVE-2026-50269/54274/54277/54279/54280 et al: fix >= 3.14.1
    assert Version(version("aiohttp")) >= Version("3.14.1")


def test_bleach_not_vulnerable_ghsa_gj48_438w_jh9v() -> None:
    # GHSA-gj48-438w-jh9v/GHSA-8rfp-98v4-mmr6: XSS/formaction bypass; fix >= 6.4.0
    assert Version(version("bleach")) >= Version("6.4.0")


def test_langchain_not_vulnerable_ghsa_gr75_jv2w_4656() -> None:
    # GHSA-gr75-jv2w-4656: path traversal in file-search middleware; fix >= 1.3.9
    assert Version(version("langchain")) >= Version("1.3.9")


@pytest.mark.xfail(
    reason=(
        "STOP condition — CVE-2026-41488 fix (langchain-openai>=1.1.14) requires openai>=2.26.0 "
        "but this repo pins openai<2.0 (breaking API change). Upgrade blocked pending openai 2.x migration. "
        "This test documents the desired state, not the achieved state."
    ),
    strict=True,
)
def test_langchain_openai_not_vulnerable_cve_2026_41488() -> None:
    # CVE-2026-41488: SSRF via unsanitized URL in _url_to_size(); fix >= 1.1.14
    assert Version(version("langchain-openai")) >= Version("1.1.14")


def test_langgraph_sdk_not_vulnerable_cve_2026_48776() -> None:
    # CVE-2026-48776: path traversal/SSRF in HTTP request path construction; fix >= 0.3.15
    assert Version(version("langgraph-sdk")) >= Version("0.3.15")


def test_pyarrow_not_vulnerable_ghsa_rgxp_2hwp_jwgg() -> None:
    # GHSA-rgxp-2hwp-jwgg / CVE-2026-25087: use-after-free in IPC pre-buffering; fix >= 23.0.1
    assert Version(version("pyarrow")) >= Version("23.0.1")


def test_starlette_not_vulnerable_cve_2026_48710() -> None:
    # Multiple starlette CVEs (CVE-2026-48710/48817/48818, PYSEC-2026-248/249); fix >= 1.3.1
    assert Version(version("starlette")) >= Version("1.3.1")


def test_pydantic_settings_not_vulnerable_ghsa_4xgf_cpjx_pc3j() -> None:
    # GHSA-4xgf-cpjx-pc3j: symlink follow outside secrets_dir; fix >= 2.14.2
    assert Version(version("pydantic-settings")) >= Version("2.14.2")
