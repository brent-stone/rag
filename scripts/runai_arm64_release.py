#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for publishing a RUN:AI-compatible ARM64 RAG release."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


SAFE_PREFIX = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
SAFE_OWNER = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?")
SAFE_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
SEMVER = re.compile(r"\d+\.\d+\.\d+")
SHA256 = re.compile(r"^Digest:\s+(sha256:[0-9a-f]{64})$", re.MULTILINE)


@dataclass(frozen=True)
class ReleaseMetadata:
    release_tag: str
    chart_version: str
    registry: str
    chart_filename: str


@dataclass(frozen=True)
class AuditResult:
    image: str
    status: str
    digest: str
    detail: str


def build_release_metadata(
    prefix: str,
    run_number: str,
    run_attempt: str,
    owner: str,
    base_version: str = "2.6.0",
) -> ReleaseMetadata:
    """Validate workflow inputs and derive immutable release identifiers."""

    if not SAFE_PREFIX.fullmatch(prefix) or ".." in prefix:
        raise ValueError(
            "release prefix must be lowercase, start with an alphanumeric "
            "character, and contain only letters, digits, '.', '_' or '-'"
        )
    if not run_number.isdigit() or int(run_number) < 1:
        raise ValueError("run number must be a positive integer")
    if not run_attempt.isdigit() or int(run_attempt) < 1:
        raise ValueError("run attempt must be a positive integer")
    if not SEMVER.fullmatch(base_version):
        raise ValueError("base version must use MAJOR.MINOR.PATCH")

    namespace = owner.lower()
    if not SAFE_OWNER.fullmatch(namespace):
        raise ValueError("repository owner is not a valid GHCR namespace")

    release_tag = f"{prefix}-{run_number}-{run_attempt}"
    chart_version = (
        f"{base_version}-runai.arm64.{run_number}.{run_attempt}"
    )
    if not SAFE_TAG.fullmatch(release_tag):
        raise ValueError("derived release tag is not a valid OCI tag")

    return ReleaseMetadata(
        release_tag=release_tag,
        chart_version=chart_version,
        registry=f"ghcr.io/{namespace}",
        chart_filename=f"nvidia-blueprint-rag-{chart_version}.tgz",
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _required_mapping(
    parent: dict[str, Any], key: str, context: str
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def customize_chart(
    chart_dir: Path,
    registry: str,
    release_tag: str,
    chart_version: str,
) -> None:
    """Bind a temporary chart copy to the four images from one workflow run."""

    if not registry.startswith("ghcr.io/"):
        raise ValueError("registry must be a ghcr.io namespace")
    if not SAFE_TAG.fullmatch(release_tag):
        raise ValueError("release tag is not a valid OCI tag")
    if not re.fullmatch(r"\d+\.\d+\.\d+-runai\.arm64\.\d+\.\d+", chart_version):
        raise ValueError("chart version is not a RUN:AI ARM64 release version")

    values_path = chart_dir / "values.yaml"
    chart_path = chart_dir / "Chart.yaml"
    values = _load_yaml_mapping(values_path)
    chart = _load_yaml_mapping(chart_path)
    registry = registry.rstrip("/")

    image_targets = (
        (("image",), "rag-server"),
        (("ingestor-server", "image"), "ingestor-server"),
        (("frontend", "image"), "rag-frontend"),
        (("nv-ingest", "image"), "nv-ingest"),
    )
    for path, image_name in image_targets:
        current = values
        context = "values"
        for key in path:
            current = _required_mapping(current, key, context)
            context = f"{context}.{key}"
        current["repository"] = f"{registry}/{image_name}"
        current["tag"] = release_tag

    chart["version"] = chart_version
    chart["appVersion"] = f"v{chart_version}"

    values_path.write_text(
        yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
    )
    chart_path.write_text(
        yaml.safe_dump(chart, sort_keys=False), encoding="utf-8"
    )


def collect_enabled_nim_images(values: Any) -> list[str]:
    """Discover enabled nvcr.io/nim images in nested Helm values."""

    images: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            image = node.get("image")
            if node.get("enabled") is True and isinstance(image, dict):
                repository = image.get("repository")
                tag = image.get("tag")
                if (
                    isinstance(repository, str)
                    and repository.startswith("nvcr.io/nim/")
                    and isinstance(tag, (str, int, float))
                    and str(tag)
                ):
                    images.add(f"{repository}:{tag}")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(values)
    return sorted(images)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _inspect_json(image: str, field: str, runner: Runner) -> dict[str, Any]:
    completed = runner(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "--format",
            f"{{{{json .{field}}}}}",
            image,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"Buildx returned no {field} object")
    return value


def audit_image(
    image: str, runner: Runner = subprocess.run
) -> AuditResult:
    """Classify one remote image by its published platform metadata."""

    try:
        description = runner(
            ["docker", "buildx", "imagetools", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
        )
        digest_match = SHA256.search(description.stdout)
        digest = digest_match.group(1) if digest_match else ""
        manifest = _inspect_json(image, "Manifest", runner)

        platforms: set[str] = set()
        descriptors = manifest.get("manifests")
        if isinstance(descriptors, list) and descriptors:
            for descriptor in descriptors:
                if not isinstance(descriptor, dict):
                    continue
                platform = descriptor.get("platform")
                if not isinstance(platform, dict):
                    continue
                os_name = platform.get("os")
                architecture = platform.get("architecture")
                if isinstance(os_name, str) and isinstance(
                    architecture, str
                ):
                    platforms.add(f"{os_name}/{architecture}")
        else:
            config = _inspect_json(image, "Image", runner)
            os_name = config.get("os")
            architecture = config.get("architecture")
            if isinstance(os_name, str) and isinstance(architecture, str):
                platforms.add(f"{os_name}/{architecture}")

        if not platforms:
            raise ValueError("registry response did not identify any platform")
        if "linux/arm64" in platforms:
            return AuditResult(
                image,
                "ARM64_AVAILABLE",
                digest,
                ", ".join(sorted(platforms)),
            )
        return AuditResult(
            image,
            "ARM64_MISSING",
            digest,
            ", ".join(sorted(platforms)),
        )
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = error.stderr.strip()
        return AuditResult(
            image,
            "INSPECTION_FAILED",
            "",
            detail.replace("\n", " ")[:500],
        )


def format_audit_report(results: Sequence[AuditResult]) -> str:
    lines = [
        "| Image reference | Status | Platforms / error |",
        "| --- | --- | --- |",
    ]
    for result in results:
        reference = (
            f"{result.image}@{result.digest}" if result.digest else result.image
        )
        detail = result.detail.replace("|", "\\|")
        lines.append(f"| `{reference}` | {result.status} | {detail} |")
    return "\n".join(lines) + "\n"


def _metadata_command(args: argparse.Namespace) -> int:
    metadata = build_release_metadata(
        args.prefix,
        args.run_number,
        args.run_attempt,
        args.owner,
        args.base_version,
    )
    outputs = {
        "release_tag": metadata.release_tag,
        "chart_version": metadata.chart_version,
        "registry": metadata.registry,
        "chart_filename": metadata.chart_filename,
    }
    with args.github_output.open("a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


def _customize_chart_command(args: argparse.Namespace) -> int:
    customize_chart(
        args.chart_dir,
        args.registry,
        args.release_tag,
        args.chart_version,
    )
    print(f"Customized chart in {args.chart_dir}")
    return 0


def _audit_command(args: argparse.Namespace) -> int:
    images = set(args.image)
    if args.values is not None:
        values = _load_yaml_mapping(args.values)
        images.update(collect_enabled_nim_images(values))
    if not images:
        raise ValueError("no images were supplied or discovered")

    results = [audit_image(image) for image in sorted(images)]
    report = format_audit_report(results)
    args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")
    return (
        0
        if all(result.status == "ARM64_AVAILABLE" for result in results)
        else 1
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser(
        "metadata", help="derive immutable workflow release identifiers"
    )
    metadata.add_argument("--prefix", required=True)
    metadata.add_argument("--run-number", required=True)
    metadata.add_argument("--run-attempt", required=True)
    metadata.add_argument("--owner", required=True)
    metadata.add_argument("--base-version", default="2.6.0")
    metadata.add_argument(
        "--github-output", required=True, type=Path
    )
    metadata.set_defaults(handler=_metadata_command)

    customize = subparsers.add_parser(
        "customize-chart", help="bind a temporary chart to published images"
    )
    customize.add_argument("--chart-dir", required=True, type=Path)
    customize.add_argument("--registry", required=True)
    customize.add_argument("--release-tag", required=True)
    customize.add_argument("--chart-version", required=True)
    customize.set_defaults(handler=_customize_chart_command)

    audit = subparsers.add_parser(
        "audit", help="require linux/arm64 manifests for remote images"
    )
    audit.add_argument("--image", action="append", default=[])
    audit.add_argument("--values", type=Path)
    audit.add_argument("--markdown-output", required=True, type=Path)
    audit.set_defaults(handler=_audit_command)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
