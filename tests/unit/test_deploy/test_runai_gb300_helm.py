# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tarfile

import yaml

from scripts.runai_arm64_release import deep_merge, patch_nv_ingest_dependency


REPO_ROOT = Path(__file__).resolve().parents[3]
CHART_DIR = REPO_ROOT / "deploy" / "helm" / "nvidia-blueprint-rag"
PROFILE_PATH = CHART_DIR / "values-runai-gb300.yaml"
DEFAULT_VALUES_PATH = CHART_DIR / "values.yaml"
INGRESS_TEMPLATE_PATH = CHART_DIR / "templates" / "ingress.yaml"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "publish-runai-arm64.yml"

ULTRA_URL = (
    "http://orchestrator-nemo3-ultra-550b-stone-g320."
    "runai-nccl.svc.cluster.local/v1"
)
SUPER_URL = (
    "http://planner-nemo3-super-120b-stone-g320."
    "runai-nccl.svc.cluster.local/v1"
)
NANO_URL = (
    "http://ingest-nemo3-nano-omni-stone-g320."
    "runai-nccl.svc.cluster.local/v1"
)
ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NANO_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def test_runai_profile_routes_llm_roles_to_shared_inference_services() -> None:
    values = _load_yaml(PROFILE_PATH)
    env = values["envVars"]
    ingestor_env = values["ingestor-server"]["envVars"]

    ultra_roles = (
        ("APP_LLM_MODELNAME", "APP_LLM_SERVERURL"),
        ("AGENTIC_PLANNER_LLM_MODEL", "AGENTIC_PLANNER_LLM_SERVERURL"),
        ("AGENTIC_SYNTHESIS_LLM_MODEL", "AGENTIC_SYNTHESIS_LLM_SERVERURL"),
    )
    for model_key, url_key in ultra_roles:
        assert env[model_key] == ULTRA_MODEL
        assert env[url_key] == ULTRA_URL

    super_roles = (
        ("APP_QUERYREWRITER_MODELNAME", "APP_QUERYREWRITER_SERVERURL"),
        (
            "APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
            "APP_FILTEREXPRESSIONGENERATOR_SERVERURL",
        ),
        ("REFLECTION_LLM", "REFLECTION_LLM_SERVERURL"),
        ("AGENTIC_TASK_LLM_MODEL", "AGENTIC_TASK_LLM_SERVERURL"),
        ("AGENTIC_SEED_GEN_LLM_MODEL", "AGENTIC_SEED_GEN_LLM_SERVERURL"),
    )
    for model_key, url_key in super_roles:
        assert env[model_key] == SUPER_MODEL
        assert env[url_key] == SUPER_URL

    assert ingestor_env["SUMMARY_LLM"] == SUPER_MODEL
    assert ingestor_env["SUMMARY_LLM_SERVERURL"] == SUPER_URL
    assert env["APP_VLM_MODELNAME"] == NANO_MODEL
    assert env["APP_VLM_SERVERURL"] == NANO_URL
    assert ingestor_env["APP_NVINGEST_CAPTIONMODELNAME"] == NANO_MODEL
    assert ingestor_env["APP_NVINGEST_CAPTIONENDPOINTURL"] == (
        f"{NANO_URL}/chat/completions"
    )
    assert values["nimOperator"]["nim-llm"]["enabled"] is False


def test_runai_profile_keeps_six_chart_managed_nims() -> None:
    values = _load_yaml(PROFILE_PATH)
    nim = values["nimOperator"]
    nv_ingest_nim = values["nv-ingest"]["nimOperator"]

    top_level_nims = (
        "nvidia-nim-llama-nemotron-embed-vl-1b-v2",
        "nvidia-nim-llama-nemotron-rerank-1b-v2",
    )
    extraction_nims = ("ocr", "page_elements", "graphic_elements", "table_structure")

    gpu_total = 0
    for name in top_level_nims:
        assert nim[name]["enabled"] is True
        gpu_total += nim[name]["resources"]["limits"]["nvidia.com/gpu"]

    for name in extraction_nims:
        config = nv_ingest_nim[name]
        assert config["enabled"] is True
        assert config["authSecret"] == "genericsecret-aiq-credentials"
        assert config["image"]["pullSecrets"] == ["dockerregistry-ngc-secret"]
        gpu_total += config["resources"]["limits"]["nvidia.com/gpu"]

    assert gpu_total == 6


def test_runai_profile_references_existing_secrets_and_ingresses() -> None:
    values = _load_yaml(PROFILE_PATH)

    assert values["imagePullSecret"] == {
        "name": "dockerregistry-ngc-secret",
        "create": False,
    }
    assert values["ingestor-server"]["imagePullSecret"]["name"] == (
        "dockerregistry-ngc-secret"
    )
    assert values["frontend"]["imagePullSecret"]["name"] == (
        "dockerregistry-ngc-secret"
    )
    assert values["nv-ingest"]["imagePullSecrets"] == [
        {"name": "dockerregistry-ngc-secret"}
    ]
    assert values["ngcApiSecret"] == {
        "name": "genericsecret-aiq-credentials",
        "create": False,
    }
    assert values["nv-ingest"]["ngcApiSecret"] == {
        "name": "genericsecret-aiq-credentials",
        "create": False,
    }
    assert values["apiKeysSecret"]["create"] is False
    assert values["apiKeysSecret"]["existingSecret"] == ""

    assert values["service"]["type"] == "ClusterIP"
    assert values["ingress"]["enabled"] is True
    assert values["ingress"]["className"] == "nginx"
    assert values["ingress"]["hosts"][0]["host"] == (
        "nccl-rag-backend.runai.ai.nps.edu"
    )

    frontend = values["frontend"]
    assert frontend["service"]["type"] == "ClusterIP"
    assert frontend["ingress"]["enabled"] is True
    assert frontend["ingress"]["className"] == "nginx"
    assert frontend["ingress"]["hosts"][0]["host"] == (
        "nccl-rag-frontend.runai.ai.nps.edu"
    )


def test_ingresses_are_opt_in_in_default_values() -> None:
    values = _load_yaml(DEFAULT_VALUES_PATH)

    assert values["ingress"]["enabled"] is False
    assert values["frontend"]["ingress"]["enabled"] is False


def test_ingress_template_routes_backend_and_frontend_services() -> None:
    template = INGRESS_TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "if .Values.ingress.enabled" in template
    assert 'include "nvidia-blueprint-rag.fullname" .' in template
    assert ".Values.service.port" in template
    assert "if .Values.frontend.ingress.enabled" in template
    assert ".Values.frontend.appName" in template
    assert ".Values.frontend.service.port" in template


def test_arm64_release_validates_and_publishes_runai_values() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'runai_values="$release_dir/values-runai-gb300.yaml"' in workflow
    assert 'helm lint "$CHART_DIR" -f "$RUNAI_VALUES"' in workflow
    assert "patch-nv-ingest-chart" in workflow
    assert 'effective_values="$release_dir/effective-values.yaml"' in workflow
    assert '--values "${{ steps.workspace.outputs.effective_values }}"' in workflow
    assert (
        'helm template rag "$CHART_DIR" --namespace runai-nccl '
        '--api-versions apps.nvidia.com/v1alpha1 -f "$RUNAI_VALUES"' in workflow
    )
    assert 'ngc_secret_names == {"genericsecret-aiq-credentials"}' in workflow
    assert "assert len(nim_services) == 6" in workflow
    assert '"$RELEASE_DIR/values-runai-gb300.yaml"' in workflow
    assert "values_url=" in workflow
    assert 'echo "- RUN:AI values: $values_url"' in workflow


def test_deep_merge_builds_effective_values_without_mutating_inputs() -> None:
    base = {
        "nimOperator": {
            "nim-llm": {"enabled": True, "image": {"repository": "llm"}},
            "embed": {"enabled": True, "image": {"repository": "embed"}},
        },
        "list": ["base"],
    }
    overlay = {
        "nimOperator": {"nim-llm": {"enabled": False}},
        "list": ["overlay"],
    }

    merged = deep_merge(base, overlay)

    assert merged["nimOperator"]["nim-llm"] == {
        "enabled": False,
        "image": {"repository": "llm"},
    }
    assert merged["nimOperator"]["embed"]["image"]["repository"] == "embed"
    assert merged["list"] == ["overlay"]
    assert base["nimOperator"]["nim-llm"]["enabled"] is True


def test_patch_nv_ingest_dependency_makes_runtime_secret_configurable(
    tmp_path: Path,
) -> None:
    chart_dir = tmp_path / "parent"
    dependency_root = tmp_path / "source" / "nv-ingest"
    templates = dependency_root / "templates"
    templates.mkdir(parents=True)
    (templates / "deployment.yaml").write_text(
        "secretKeyRef:\n                  name: ngc-api\n"
        "                  key: NGC_API_KEY\n",
        encoding="utf-8",
    )
    (templates / "secrets.yaml").write_text(
        "metadata:\n  name: ngc-api  # Name expected by NIMs\n",
        encoding="utf-8",
    )
    (dependency_root / "values.yaml").write_text(
        "ngcApiSecret:\n  create: false\n  password: \"\"\n",
        encoding="utf-8",
    )
    charts_dir = chart_dir / "charts"
    charts_dir.mkdir(parents=True)
    archive = charts_dir / "nv-ingest-1.0.0.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(dependency_root, arcname="nv-ingest")

    patch_nv_ingest_dependency(chart_dir)

    extracted = tmp_path / "extracted"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")
    deployment = (extracted / "nv-ingest" / "templates" / "deployment.yaml").read_text()
    secret = (extracted / "nv-ingest" / "templates" / "secrets.yaml").read_text()
    dependency_values = yaml.safe_load(
        (extracted / "nv-ingest" / "values.yaml").read_text()
    )

    configurable_name = '{{ .Values.ngcApiSecret.name | default "ngc-api" }}'
    assert f"name: {configurable_name}" in deployment
    assert f"name: {configurable_name}" in secret
    assert dependency_values["ngcApiSecret"]["name"] == "ngc-api"
