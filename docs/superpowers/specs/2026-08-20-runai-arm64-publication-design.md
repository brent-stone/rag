# RUN:AI ARM64 Public Artifact Publication Design

## Purpose

Add a fork-friendly, manually triggered GitHub Actions release path that builds the RAG Blueprint application containers and its baseline NV-Ingest service for `linux/arm64`, publishes them from `brent-stone/rag`, packages a Helm chart that references those exact images, and exposes the resulting artifacts through public URLs suitable for a RUN:AI deployment test on NVIDIA DGX GB300.

This release path is additive. It does not replace or alter NVIDIA's existing NGC publication workflow.

## Scope

The workflow builds the images owned by this repository:

- `rag-server`
- `ingestor-server`
- `rag-frontend`

It also checks out NVIDIA's public NV-Ingest `26.3.0` release at immutable commit `b1aa9729809bd46c0b4f089ccd0ead946303eba3`, builds its upstream `runtime` target for ARM64 without source modifications, and publishes that image with the same run-specific tag.

It packages `deploy/helm/nvidia-blueprint-rag` after replacing all four image repositories and tags with the images produced by the same workflow run.

The workflow audits, but does not rebuild, the remaining external NVIDIA NIM images. A missing `linux/arm64` manifest in a required external image is reported as an external compatibility blocker.

## Publication Targets

Images are published to GitHub Container Registry under the fork owner:

```text
ghcr.io/brent-stone/rag-server:<release-tag>
ghcr.io/brent-stone/ingestor-server:<release-tag>
ghcr.io/brent-stone/rag-frontend:<release-tag>
ghcr.io/brent-stone/nv-ingest:<release-tag>
```

The workflow derives `<release-tag>` from a sanitized manual input and the GitHub Actions run number so that a rerun cannot overwrite an earlier test accidentally. The default input is `runai-arm64`, producing tags in the form:

```text
runai-arm64-<run-number>
```

The packaged Helm chart is attached to a GitHub Release in the public `brent-stone/rag` repository. The release tag uses the same image tag, and the chart version is a valid SemVer prerelease in the form:

```text
2.6.0-runai.arm64.<run-number>
```

After publication, the workflow summary prints:

- the four immutable image references, including registry digests;
- the GitHub Release page URL;
- the direct chart download URL;
- an example `helm upgrade --install` command;
- the ARM64 compatibility audit result for external images.

## Trigger and Permissions

The workflow is `workflow_dispatch` only. It can be run from the feature branch in the fork and accepts:

- `release_prefix`, default `runai-arm64`;
- `audit_external_images`, default `true`.

The workflow grants only:

```yaml
permissions:
  contents: write
  packages: write
```

`GITHUB_TOKEN` authenticates to GHCR, checks out the public NV-Ingest source, and creates the GitHub Release. `NGC_API_KEY` is a repository secret used only to authenticate to NGC when resolving Helm dependencies and inspecting NVIDIA image manifests. An optional `HF_ACCESS_TOKEN` is passed to the NV-Ingest build as a BuildKit secret for its upstream tokenizer download step. Secret values are never printed.

## Native ARM64 Build

Container builds run on GitHub's native `ubuntu-24.04-arm` hosted runner. Each build explicitly requests `linux/arm64` through `docker/build-push-action` and pushes directly to GHCR.

Native ARM execution is preferred over QEMU because the Python environments include compiled dependencies and the images are large. The existing Dockerfiles remain multi-stage builds: all stages resolve for the requested target platform, so the copied Python executable and `/workspace/.venv` are ARM64 artifacts.

The NV-Ingest job uses NVIDIA's unmodified `26.3.0` Dockerfile and mirrors the build contract in that release's ARM workflow:

```text
target: runtime
platform: linux/arm64
BASE_IMG: ubuntu
BASE_IMG_TAG: jammy-20250415.1
DOWNLOAD_LLAMA_TOKENIZER: True
GIT_COMMIT: b1aa9729809bd46c0b4f089ccd0ead946303eba3
```

The immutable commit, upstream repository URL, release tag, and Apache-2.0 license are recorded in OCI labels and the GitHub Release notes. A copy of the upstream license is attached to the release. The upstream Dockerfile's bundled GPL-source handling is retained unchanged.

Each image build includes OCI source and revision labels linking the package to its public source repository. After pushing, the workflow checks the published manifest and verifies that it contains `linux/arm64` before allowing Helm packaging to proceed.

## Helm Packaging

Helm dependencies are resolved in a temporary working copy of the chart. The committed default `values.yaml` remains pointed at NVIDIA's release images.

Before packaging, the workflow changes only these values in the temporary chart:

```yaml
image.repository: ghcr.io/brent-stone/rag-server
image.tag: <release-tag>
ingestor-server.image.repository: ghcr.io/brent-stone/ingestor-server
ingestor-server.image.tag: <release-tag>
frontend.image.repository: ghcr.io/brent-stone/rag-frontend
frontend.image.tag: <release-tag>
nv-ingest.image.repository: ghcr.io/brent-stone/nv-ingest
nv-ingest.image.tag: <release-tag>
```

It also sets the chart version and `appVersion` to the run-specific release version. The packaged chart is linted with its resolved dependencies before it is attached to the release.

The packaged chart continues to create and use the existing `ngc-secret` for NVIDIA-hosted dependencies. Public GHCR application and NV-Ingest images require no GHCR pull secret once their packages inherit the public repository visibility. The workflow performs an anonymous manifest inspection after logging out of GHCR; publication fails with a package-visibility instruction if anonymous access is unavailable.

## External ARM64 Compatibility Audit

The audit checks the enabled NVIDIA NIM images used by the chart. NV-Ingest is excluded from this external audit because its published ARM64 manifest is already verified by the same release workflow. For each external image the audit records one of:

- `ARM64_AVAILABLE`: the registry exposes a `linux/arm64` manifest;
- `ARM64_MISSING`: the registry resolves the tag but exposes no ARM64 manifest;
- `INSPECTION_FAILED`: authentication, tag resolution, or registry access failed.

`ARM64_MISSING` is a hard failure because publishing an installable-looking chart would reproduce `exec format error` on GB300. `INSPECTION_FAILED` is also a failure when `audit_external_images` is true because compatibility has not been established. Setting the input to false permits diagnostic publication of the four ARM64 images and chart, but the workflow summary prominently marks external compatibility as unverified.

## Failure Handling

The release job must stop before creating a GitHub Release when any of these conditions occurs:

- an application or NV-Ingest image fails to build;
- a pushed image manifest lacks `linux/arm64`;
- a required external image fails the enabled ARM64 audit;
- Helm dependency resolution fails;
- Helm lint fails;
- anonymous access to the published GHCR images fails.

Build artifacts may exist in GHCR after a later gate fails. The summary reports any successfully pushed image references so they can be inspected or deleted manually. The workflow does not delete or overwrite prior packages or releases.

## Testing Strategy

Repository tests validate the workflow and release helper behavior without pushing artifacts:

1. The workflow is manual-only and grants the required minimal permissions.
2. All four build jobs use `ubuntu-24.04-arm`, request `linux/arm64`, and target GHCR.
3. Release tag and chart-version normalization reject invalid or unsafe input.
4. The NV-Ingest build is pinned to the approved `26.3.0` commit and upstream ARM build arguments.
5. The chart customization helper changes only the intended repositories, tags, and version fields.
6. The external-image audit distinguishes available, missing, and failed manifest inspections.
7. A locally packaged chart passes `helm lint` when dependencies are available.

The GitHub workflow supplies the integration evidence that cannot be produced locally: native ARM64 image builds, published manifest inspection, anonymous pulls, and public release URLs.

## RUN:AI Retest Contract

Once the workflow succeeds, the operator downloads or installs the chart from the emitted public release URL and verifies the cluster architecture before deployment:

```bash
kubectl get nodes -L kubernetes.io/arch
helm upgrade --install rag <public-chart-url> \
  --namespace nv-nvidia-blueprint-rag \
  --create-namespace \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY"
```

The post-deployment check is:

```bash
kubectl get pods -n nv-nvidia-blueprint-rag -o wide
kubectl get events -n nv-nvidia-blueprint-rag --sort-by='.lastTimestamp'
kubectl logs -n nv-nvidia-blueprint-rag deployment/rag-server --tail=100
kubectl logs -n nv-nvidia-blueprint-rag deployment/ingestor-server --tail=100
kubectl logs -n nv-nvidia-blueprint-rag deployment/rag-nv-ingest --tail=100
```

A successful baseline retest means the three repository-owned services and NV-Ingest no longer fail with `exec format error`. Overall RAG readiness additionally requires every audited NVIDIA dependency to expose a working ARM64 image and all ordinary health checks to pass.

## Non-Goals

- Modifying or maintaining a fork of the NV-Ingest source.
- Rebuilding or redistributing NVIDIA NIM containers.
- Changing the existing NGC release workflow.
- Claiming GB300 support for external NVIDIA images that cannot be inspected.
- Adding runtime CPU emulation to Kubernetes nodes.
- Automatically deploying to or mutating the RUN:AI cluster from GitHub Actions.
