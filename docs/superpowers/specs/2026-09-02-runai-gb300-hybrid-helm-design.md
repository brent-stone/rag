# RUN:AI GB300 Hybrid Helm Deployment Design

## Goal

Publish an ARM64 NVIDIA RAG Blueprint Helm chart and a RUN:AI values file that deploy successfully in the `runai-nccl` namespace on the DGX GB300 cluster.

## Deployment Architecture

The RAG server uses existing RUN:AI inference services for language-model roles while Helm manages the default VLM embedder, text reranker, OCR, page-elements, graphic-elements, and table-structure NIMs. The chart-managed inference tier therefore requests six GPUs. Elasticsearch, SeaweedFS, Redis, NV-Ingest, the ingestor server, RAG server, and frontend remain chart-managed.

The main generation, agentic planner, and agentic synthesis roles use the existing Nemotron Ultra service. Query rewriting, filter generation, reflection, ingestion summarization, agentic task execution, and agentic seed generation use the existing Nemotron Super service. The existing Nano Omni service is preconfigured for optional VLM generation and ingestion captioning. The chart-managed LLM NIM is disabled to avoid duplicating those services.

## RUN:AI Integration

- Namespace: `runai-nccl` (selected by the RUN:AI AI Applications deployment).
- NVIDIA API secret: `genericsecret-aiq-credentials`, with `NVIDIA_API_KEY` and `NGC_API_KEY` keys.
- NGC image pull secret: `dockerregistry-ngc-secret`, which is a verified `kubernetes.io/dockerconfigjson` secret containing an `nvcr.io` login in `runai-nccl`.
- Ingress class: `nginx`.
- Frontend hostname: `nccl-rag-frontend.runai.ai.nps.edu`.
- RAG API hostname: `nccl-rag-backend.runai.ai.nps.edu`.
- Persistent volumes use the cluster default StorageClass and `ReadWriteOnce` access.

## External Model Endpoints

- Ultra URL: `http://orchestrator-nemo3-ultra-550b-stone-g320.runai-nccl.svc.cluster.local/v1`
- Ultra model: `nvidia/nemotron-3-ultra-550b-a55b`
- Super URL: `http://planner-nemo3-super-120b-stone-g320.runai-nccl.svc.cluster.local/v1`
- Super model: `nvidia/nemotron-3-super-120b-a12b`
- Nano Omni URL: `http://ingest-nemo3-nano-omni-stone-g320.runai-nccl.svc.cluster.local/v1`
- Nano Omni model: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`

## Chart Changes

The chart gains optional Kubernetes Ingress resources for the RAG server and frontend. They are disabled by default so upstream behavior remains unchanged. The RUN:AI overlay enables both ingresses and converts the frontend service from `NodePort` to `ClusterIP`.

The RUN:AI overlay explicitly redirects every LLM role, disables only the duplicate chart-managed LLM, keeps the six key chart-managed inference NIMs enabled, and overrides NV-Ingest subchart authentication references so RUN:AI's `genericsecret-` name transformation reaches every NIMService.

## Release Contract

The ARM64 publication workflow must lint the chart with the RUN:AI overlay, include the overlay as a standalone GitHub release asset, and package the overlay inside the Helm chart. The release continues to publish ARM64 RAG server, ingestor, frontend, and NV-Ingest images and audit enabled NVIDIA NIM images for ARM64 support.

## Verification

Automated tests validate the role-to-model mapping, secret propagation, six-GPU NIM selection, ingress configuration, and release workflow asset handling. Helm lint and template rendering validate the packaged chart with resolved dependencies before publication.
