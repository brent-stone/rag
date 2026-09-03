# RUN:AI GB300 Hybrid Helm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an ARM64 Helm chart and values overlay for a six-GPU hybrid NVIDIA RAG deployment on RUN:AI DGX GB300.

**Architecture:** Optional chart-level nginx ingresses expose the frontend and RAG API. A dedicated overlay references RUN:AI secrets, routes language roles to existing Ultra and Super services, routes optional VLM/captioning calls to the existing Nano Omni service, and retains the chart-managed VLM embedder, reranker, and four extraction NIMs.

**Tech Stack:** Helm 3/4, Kubernetes, NVIDIA NIM Operator, NV-Ingest, GitHub Actions, Python/PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-runai-gb300-hybrid-helm-design.md`

## Global Constraints

- Preserve the existing native `linux/arm64` image build workflow.
- Keep existing chart behavior unchanged unless the RUN:AI overlay is supplied.
- Never place API-key values in Git or Helm values.
- Use `genericsecret-aiq-credentials` for NVIDIA API-key references and the verified `dockerregistry-ngc-secret` for NGC image pulls.
- Keep all six chart-managed embedding, reranking, and extraction NIMs enabled.

---

### Task 1: Define RUN:AI Profile Contract Tests

**Files:**
- Create: `tests/unit/test_deploy/test_runai_gb300_helm.py`
- Create: `deploy/helm/nvidia-blueprint-rag/values-runai-gb300.yaml`

**Interfaces:**
- Consumes: Helm value keys already defined in `values.yaml`.
- Produces: assertions for secrets, model roles, enabled NIMs, GPU total, and ingress hosts.

- [ ] Write tests that load `values-runai-gb300.yaml` and assert the exact deployment contract.
- [ ] Run `pytest -q tests/unit/test_deploy/test_runai_gb300_helm.py` and verify failure because the overlay is absent.
- [ ] Add the minimal overlay with exact RUN:AI secret names, external role URLs, and six enabled one-GPU NIMs.
- [ ] Re-run the profile tests and verify they pass.

### Task 2: Add Optional Ingress Rendering

**Files:**
- Modify: `deploy/helm/nvidia-blueprint-rag/values.yaml`
- Create: `deploy/helm/nvidia-blueprint-rag/templates/ingress.yaml`
- Modify: `tests/unit/test_deploy/test_runai_gb300_helm.py`

**Interfaces:**
- Consumes: `ingress` and `frontend.ingress` values.
- Produces: networking.k8s.io/v1 Ingress resources targeting `rag-server:8081` and `rag-frontend:3000`.

- [ ] Add tests that require default-disabled ingress values and both target services in the template.
- [ ] Run the focused tests and verify the new ingress assertions fail.
- [ ] Add default ingress values and the two conditional ingress documents.
- [ ] Run the focused tests and verify they pass.

### Task 3: Publish the RUN:AI Overlay

**Files:**
- Modify: `.github/workflows/publish-runai-arm64.yml`
- Modify: `tests/unit/test_deploy/test_runai_gb300_helm.py`

**Interfaces:**
- Consumes: `values-runai-gb300.yaml` from the customized chart workspace.
- Produces: a standalone release asset named `values-runai-gb300.yaml` and Helm lint/template checks using it.

- [ ] Add workflow contract assertions for copying, linting, templating, uploading, and linking the overlay.
- [ ] Run the focused tests and verify the workflow assertions fail.
- [ ] Update the publication workflow with the required overlay handling.
- [ ] Run the focused tests and verify they pass.

### Task 4: Verify, Commit, and Publish

**Files:**
- Verify all files changed by Tasks 1-3.

**Interfaces:**
- Consumes: complete feature branch.
- Produces: signed-off commit, pushed branch, successful GitHub Actions release, chart URL, and values URL.

- [ ] Run the focused pytest suite.
- [ ] Resolve Helm dependencies in a temporary chart copy and run `helm lint` plus `helm template` with the RUN:AI overlay.
- [ ] Review `git diff --check`, the staged diff, and secret-value absence.
- [ ] Commit with DCO sign-off and push `feature/runai-gb300-rag-deployment`.
- [ ] Dispatch `publish-runai-arm64.yml`, wait for completion, and verify the GitHub release assets.
