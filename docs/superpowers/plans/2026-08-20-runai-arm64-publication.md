# RUN:AI ARM64 Public Artifact Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual GitHub Actions path that builds four native ARM64 images, publishes them publicly to GHCR, packages a chart bound to those images, and emits URLs and commands for a DGX GB300 RUN:AI retest.

**Architecture:** A small Python helper owns release-name validation, temporary Helm value customization, enabled-NIM discovery, and registry platform inspection. Unit tests exercise that behavior directly, while a separate structural test treats the GitHub workflow as configuration under test. The workflow fans out the three repository builds as a native ARM64 matrix, builds pinned NV-Ingest source in a fourth native job, then gates a GitHub Release on anonymous GHCR inspection, external NIM inspection, Helm dependency resolution, and Helm lint.

**Tech Stack:** Python 3.12, PyYAML, pytest, GitHub Actions, Docker Buildx, GHCR, Helm, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-08-20-runai-arm64-publication-design.md`

## Global Constraints

- Run only through `workflow_dispatch`; do not change `.github/workflows/publish-artifacts.yml`.
- Build `rag-server`, `ingestor-server`, `rag-frontend`, and `nv-ingest` for `linux/arm64` on `ubuntu-24.04-arm`.
- Pin NV-Ingest to `NVIDIA/NeMo-Retriever` commit `b1aa9729809bd46c0b4f089ccd0ead946303eba3`, release `26.3.0`.
- Build NV-Ingest target `runtime` with `BASE_IMG=ubuntu`, `BASE_IMG_TAG=jammy-20250415.1`, and `DOWNLOAD_LLAMA_TOKENIZER=True`.
- Publish under `ghcr.io/<lowercase-fork-owner>/` and verify anonymous `linux/arm64` access before release.
- Include `github.run_number` and `github.run_attempt` in tags and chart versions so workflow reruns cannot overwrite earlier artifacts.
- Modify only a temporary chart copy; keep committed default Helm values unchanged.
- Preserve NV-Ingest Apache-2.0 attribution and attach its license to the release.
- Treat missing or uninspectable enabled NIM ARM64 manifests as fatal when `audit_external_images` is true.
- Do not commit. Stage only this plan, the approved design, the helper, workflow, and focused tests; preserve all unrelated user changes.

---

### Task 1: Release metadata and chart customization

**Files:**
- Create: `scripts/runai_arm64_release.py`
- Create: `tests/unit/test_deploy/test_runai_arm64_release.py`

**Interfaces:**
- Produces: `ReleaseMetadata(release_tag, chart_version, registry)`.
- Produces: `build_release_metadata(prefix, run_number, run_attempt, owner, base_version="2.6.0") -> ReleaseMetadata`.
- Produces: `customize_chart(chart_dir: Path, registry: str, release_tag: str, chart_version: str) -> None`.
- Consumes: `values.yaml` keys `image`, `ingestor-server.image`, `frontend.image`, and `nv-ingest.image`.

- [ ] **Step 1: Write failing metadata tests**

```python
def test_build_release_metadata_is_unique_per_attempt():
    first = build_release_metadata("runai-arm64", "42", "1", "Brent-Stone")
    retry = build_release_metadata("runai-arm64", "42", "2", "Brent-Stone")
    assert first.release_tag == "runai-arm64-42-1"
    assert retry.release_tag == "runai-arm64-42-2"
    assert first.chart_version == "2.6.0-runai.arm64.42.1"
    assert first.registry == "ghcr.io/brent-stone"


@pytest.mark.parametrize("prefix", ["", "../escape", "UPPER", "has space"])
def test_build_release_metadata_rejects_unsafe_prefix(prefix):
    with pytest.raises(ValueError):
        build_release_metadata(prefix, "42", "1", "brent-stone")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py`

Expected: FAIL because `scripts/runai_arm64_release.py` does not exist.

- [ ] **Step 3: Implement metadata validation**

```python
@dataclass(frozen=True)
class ReleaseMetadata:
    release_tag: str
    chart_version: str
    registry: str


def build_release_metadata(prefix, run_number, run_attempt, owner, base_version="2.6.0"):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", prefix) or ".." in prefix:
        raise ValueError("release prefix must be a safe lowercase OCI tag prefix")
    if not run_number.isdigit() or int(run_number) < 1:
        raise ValueError("run number must be a positive integer")
    if not run_attempt.isdigit() or int(run_attempt) < 1:
        raise ValueError("run attempt must be a positive integer")
    owner = owner.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,38}", owner):
        raise ValueError("repository owner is not a valid GHCR namespace")
    return ReleaseMetadata(
        release_tag=f"{prefix}-{run_number}-{run_attempt}",
        chart_version=f"{base_version}-runai.arm64.{run_number}.{run_attempt}",
        registry=f"ghcr.io/{owner}",
    )
```

- [ ] **Step 4: Add chart customization tests**

```python
def test_customize_chart_changes_only_release_fields(tmp_path):
    chart_dir = copy_chart_fixture(tmp_path)
    customize_chart(
        chart_dir,
        "ghcr.io/brent-stone",
        "runai-arm64-42-1",
        "2.6.0-runai.arm64.42.1",
    )
    values = yaml.safe_load((chart_dir / "values.yaml").read_text())
    assert values["image"] == {"repository": "ghcr.io/brent-stone/rag-server", "tag": "runai-arm64-42-1"}
    assert values["ingestor-server"]["image"]["repository"] == "ghcr.io/brent-stone/ingestor-server"
    assert values["frontend"]["image"]["repository"] == "ghcr.io/brent-stone/rag-frontend"
    assert values["nv-ingest"]["image"]["repository"] == "ghcr.io/brent-stone/nv-ingest"
    chart = yaml.safe_load((chart_dir / "Chart.yaml").read_text())
    assert chart["version"] == "2.6.0-runai.arm64.42.1"
    assert chart["appVersion"] == "v2.6.0-runai.arm64.42.1"
```

- [ ] **Step 5: Run the chart test and verify RED**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py -k customize`

Expected: FAIL because `customize_chart` is not implemented.

- [ ] **Step 6: Implement temporary chart customization and the `metadata` / `customize-chart` CLI subcommands**

Use `yaml.safe_load` and `yaml.safe_dump(sort_keys=False)`. Require every expected mapping before mutation, write all four repositories and tags, then set `Chart.yaml.version` and `Chart.yaml.appVersion`. The `metadata` command writes `release_tag`, `chart_version`, and `registry` to the path supplied by `--github-output`.

- [ ] **Step 7: Run Task 1 tests and verify GREEN**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py`

Expected: PASS.

### Task 2: Enabled-NIM discovery and ARM64 manifest audit

**Files:**
- Modify: `scripts/runai_arm64_release.py`
- Modify: `tests/unit/test_deploy/test_runai_arm64_release.py`

**Interfaces:**
- Produces: `collect_enabled_nim_images(values: dict) -> list[str]`.
- Produces: `AuditResult(image: str, status: str, detail: str)`.
- Produces: `audit_image(image: str, runner=subprocess.run) -> AuditResult`.
- Produces: CLI `audit --image ... --values ... --markdown-output ...`, exiting nonzero unless every image is `ARM64_AVAILABLE`.

- [ ] **Step 1: Write failing enabled-image discovery tests**

```python
def test_collect_enabled_nim_images_excludes_disabled_and_nv_ingest():
    values = {
        "enabled-nim": {"enabled": True, "image": {"repository": "nvcr.io/nim/nvidia/a", "tag": "1"}},
        "disabled-nim": {"enabled": False, "image": {"repository": "nvcr.io/nim/nvidia/b", "tag": "2"}},
        "nv-ingest": {"enabled": True, "image": {"repository": "nvcr.io/nvidia/nemo-microservices/nv-ingest", "tag": "26.3.0"}},
    }
    assert collect_enabled_nim_images(values) == ["nvcr.io/nim/nvidia/a:1"]
```

- [ ] **Step 2: Run the discovery test and verify RED**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py -k enabled_nim`

Expected: FAIL because the collector is missing.

- [ ] **Step 3: Implement recursive enabled-NIM discovery**

Walk nested dictionaries and lists. Add a reference only when the same mapping has `enabled: true`, an `image.repository` beginning with `nvcr.io/nim/`, and a nonempty `image.tag`. Return sorted unique references.

- [ ] **Step 4: Write failing manifest classification tests**

```python
def test_audit_image_accepts_arm64_index(fake_runner):
    fake_runner.manifest({"manifests": [{"platform": {"os": "linux", "architecture": "arm64"}}]})
    assert audit_image("example/image:tag", fake_runner).status == "ARM64_AVAILABLE"


def test_audit_image_rejects_amd64_single_manifest(fake_runner):
    fake_runner.manifest({"mediaType": "application/vnd.oci.image.manifest.v1+json"})
    fake_runner.image({"os": "linux", "architecture": "amd64"})
    assert audit_image("example/image:tag", fake_runner).status == "ARM64_MISSING"


def test_audit_image_reports_inspection_failure(fake_runner):
    fake_runner.fail("denied")
    assert audit_image("example/image:tag", fake_runner).status == "INSPECTION_FAILED"
```

- [ ] **Step 5: Run audit tests and verify RED**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py -k audit_image`

Expected: FAIL because `audit_image` is missing.

- [ ] **Step 6: Implement Buildx inspection and audit CLI**

Run `docker buildx imagetools inspect --format '{{json .Manifest}}' IMAGE`. For an index, collect every descriptor platform. For a single manifest, additionally run the same command with `{{json .Image}}` and read the config's `os` and `architecture`. Convert process, JSON, and missing-platform errors to `INSPECTION_FAILED`; return `ARM64_MISSING` only after successful inspection proves ARM64 absent. Write a Markdown table and exit 1 for any non-available result.

- [ ] **Step 7: Run Task 2 tests and verify GREEN**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py`

Expected: PASS.

### Task 3: Manual native ARM64 publication workflow

**Files:**
- Create: `.github/workflows/publish-runai-arm64.yml`
- Create: `tests/unit/test_deploy/test_runai_arm64_workflow.py`

**Interfaces:**
- Consumes: helper CLI from Tasks 1-2.
- Produces: four GHCR images, a GitHub Release, a direct chart URL, and a workflow summary containing a RUN:AI install command.

- [ ] **Step 1: Write failing workflow structure tests**

```python
def test_workflow_is_manual_and_minimally_permissioned(workflow):
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "write", "packages": "write"}


def test_workflow_builds_all_four_targets_natively(workflow):
    app = workflow["jobs"]["build-app-images"]
    assert app["runs-on"] == "ubuntu-24.04-arm"
    assert {item["image"] for item in app["strategy"]["matrix"]["include"]} == {
        "rag-server", "ingestor-server", "rag-frontend"
    }
    nv = workflow["jobs"]["build-nv-ingest"]
    assert nv["runs-on"] == "ubuntu-24.04-arm"
    checkout = next(step for step in nv["steps"] if step.get("name") == "Checkout pinned NV-Ingest source")
    assert checkout["with"]["repository"] == "NVIDIA/NeMo-Retriever"
    assert checkout["with"]["ref"] == "b1aa9729809bd46c0b4f089ccd0ead946303eba3"
```

- [ ] **Step 2: Run workflow tests and verify RED**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_workflow.py`

Expected: FAIL because the workflow is absent.

- [ ] **Step 3: Add preparation and native build jobs**

Create a `prepare` job that invokes `metadata` with `github.run_number` and `github.run_attempt`. Add `build-app-images` as a three-entry matrix on `ubuntu-24.04-arm`, using `docker/build-push-action@v6`, `platforms: linux/arm64`, the exact existing build contexts/Dockerfiles, and `DOWNLOAD_LEGAL_COMPLIANCE=true`. Add `build-nv-ingest` on the same runner, check out the immutable NVIDIA commit, build target `runtime`, pass the approved upstream build arguments, and label the upstream source/revision/license.

- [ ] **Step 4: Add gated packaging and release job**

On `ubuntu-latest`, inspect the four GHCR tags anonymously with the helper before authenticating elsewhere. Validate `NGC_API_KEY`, log into NGC with password-stdin, and optionally audit enabled NIM images from the chart. Copy the chart into `RUNNER_TEMP`, invoke `customize-chart`, add all Helm repositories, run `helm dependency update`, `helm lint`, and `helm package`. Attach the chart, NV-Ingest license, and audit reports using `gh release create`. Append the release URL, direct chart URL, four digest-bearing image references, and RUN:AI install command to `GITHUB_STEP_SUMMARY`.

- [ ] **Step 5: Run workflow tests and verify GREEN**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_workflow.py`

Expected: PASS.

### Task 4: Full verification and staging

**Files:**
- Verify: `scripts/runai_arm64_release.py`
- Verify: `.github/workflows/publish-runai-arm64.yml`
- Verify: `tests/unit/test_deploy/test_runai_arm64_release.py`
- Verify: `tests/unit/test_deploy/test_runai_arm64_workflow.py`
- Stage: `docs/superpowers/specs/2026-08-20-runai-arm64-publication-design.md`
- Stage: `docs/superpowers/plans/2026-08-20-runai-arm64-publication.md`

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: a tested, staged change set ready for the user's fork build.

- [ ] **Step 1: Run focused Python tests**

Run: `uv run --isolated --no-project --with pytest --with pyyaml pytest -q tests/unit/test_deploy/test_runai_arm64_release.py tests/unit/test_deploy/test_runai_arm64_workflow.py`

Expected: all tests PASS with no warnings.

- [ ] **Step 2: Exercise helper CLI against a temporary chart**

Run:

```bash
tmp_dir=$(mktemp -d)
cp -a deploy/helm/nvidia-blueprint-rag "$tmp_dir/chart"
uv run --isolated --no-project --with pyyaml python scripts/runai_arm64_release.py customize-chart \
  --chart-dir "$tmp_dir/chart" \
  --registry ghcr.io/brent-stone \
  --release-tag runai-arm64-1-1 \
  --chart-version 2.6.0-runai.arm64.1.1
helm lint "$tmp_dir/chart" --skip-schema-validation
```

Expected: helper exits 0; Helm reaches only the expected missing-dependency condition if dependencies are not locally resolved.

- [ ] **Step 3: Validate workflow YAML and patch hygiene**

Run:

```bash
uv run --isolated --no-project --with pyyaml python -c 'import pathlib,yaml; yaml.safe_load(pathlib.Path(".github/workflows/publish-runai-arm64.yml").read_text())'
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 4: Stage only approved artifacts**

Run:

```bash
git add \
  .github/workflows/publish-runai-arm64.yml \
  scripts/runai_arm64_release.py \
  tests/unit/test_deploy/test_runai_arm64_release.py \
  tests/unit/test_deploy/test_runai_arm64_workflow.py \
  docs/superpowers/specs/2026-08-20-runai-arm64-publication-design.md \
  docs/superpowers/plans/2026-08-20-runai-arm64-publication.md
```

Expected: cached diff contains exactly those six paths; the user's LFS media and `.idea/` remain unstaged.
