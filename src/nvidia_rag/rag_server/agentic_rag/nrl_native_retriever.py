# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NRL-native retrieval path for agentic RAG.

Used only when ENABLE_NRL_NATIVE_RETRIEVAL=True and INGESTOR_BACKEND=nrl.
Bypasses the LangChain vectorstore wrapper (VDBRag.get_langchain_vectorstore /
retrieval_langchain) and queries NRL's nemo_retriever.retriever.Retriever
directly against LanceDB.

The nemo_retriever import is deferred to build_nrl_retriever() so that
installations without the optional `nrl` extras group
(pyproject.toml [project.optional-dependencies].nrl) do not fail at import
time when this module is imported but the flag is off.

A fresh Retriever instance is constructed per collection on every call — no
caching/pooling across requests. NRL's Retriever binds to exactly one LanceDB
table per instance (table_name is a singular str throughout
RetrieveVdbOperator/LanceDB), and the rest of this codebase already rebuilds
VDB operators fresh on every search()/generate() call (see
NvidiaRAG._prepare_vdb_op), so this matches existing conventions.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import TYPE_CHECKING, Any

from nvidia_rag.rag_server.response_generator import (
    Citations,
    SourceMetadata,
    SourceResult,
    get_object_store_operator_instance,
)

if TYPE_CHECKING:
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)

# Mirrors response_generator.prepare_citations_nrl's _NRL_TYPE_MAP — keep in
# sync. Both map the same NRL content_type vocabulary to the
# SourceResult.document_type Literal.
_NRL_TYPE_MAP: dict[str, str] = {
    "text": "text",
    "image": "image",
    "image_caption": "image",
    "chart": "chart",
    "chart_caption": "chart",
    "table": "table",
    "table_caption": "table",
    "audio": "audio",
    "infographic": "image",
    "infographic_caption": "image",
}


def build_nrl_retriever(
    *,
    config: NvidiaRAGConfig,
    collection_name: str,
    vdb_endpoint: str | None,
    top_k: int,
    enable_reranker: bool,
    reranker_model: str | None,
    reranker_endpoint: str | None,
) -> Any:
    """Construct one fresh NRL Retriever bound to a single LanceDB table.

    Deferred import: nemo_retriever is an optional dependency, only imported
    here so callers that never reach the NRL-native path (flag off, or
    INGESTOR_BACKEND != "nrl") never need it installed.
    """
    from nemo_retriever.retriever import Retriever  # deferred import

    embed_kwargs: dict[str, Any] = {
        "model_name": config.embeddings.model_name,
        "embed_invoke_url": config.embeddings.server_url,
        "api_key": config.embeddings.get_api_key() or "",
    }
    vdb_kwargs: dict[str, Any] = {
        "uri": vdb_endpoint or config.vector_store.url,
        "table_name": collection_name,
    }

    rerank_kwargs: dict[str, Any] = {}
    if enable_reranker:
        rerank_kwargs = {
            "model_name": reranker_model or config.ranking.model_name,
            # NemotronRerankActor expects "rerank_invoke_url", not
            # "endpoint"/"url"/"server_url" — verified against
            # nemo_retriever/operators/rerank.py.
            "rerank_invoke_url": reranker_endpoint or config.ranking.server_url,
            "api_key": config.ranking.get_api_key() or "",
        }

    return Retriever(
        run_mode="service",  # HTTP embed calls; no in-process Ray cluster
        top_k=top_k,
        rerank=enable_reranker,
        embed_kwargs=embed_kwargs,
        vdb_kwargs=vdb_kwargs,
        rerank_kwargs=rerank_kwargs,
    )


def _hit_to_source_result(hit: dict[str, Any], stage: str) -> SourceResult | None:
    """Map one NRL RetrievalHit -> SourceResult.

    Mirrors response_generator.prepare_citations_nrl's field-mapping
    conventions (content-type vocabulary, stored_image_uri handling,
    filename/path fallback, page_number, score fallback chain), reading from
    a RetrievalHit dict's own keys instead of a LangChain Document.metadata
    dict.
    """
    stored_image_uri: str = hit.get("stored_image_uri") or ""
    nrl_content_type = str(hit.get("content_type") or "").strip().lower()

    if stored_image_uri:
        document_type = _NRL_TYPE_MAP.get(nrl_content_type, "image")
    else:
        document_type = _NRL_TYPE_MAP.get(nrl_content_type, "text")

    content = ""
    if stored_image_uri and document_type != "text":
        try:
            raw_bytes = get_object_store_operator_instance().get_object_from_uri(
                stored_image_uri
            )
            content = base64.b64encode(raw_bytes).decode("ascii")
        except Exception:
            logger.exception(
                "[NRL Native Retriever] Failed to fetch visual asset from "
                "object storage (uri=%s)",
                stored_image_uri,
            )
            content = ""
    else:
        content = hit.get("text") or ""

    if not content:
        return None
    if document_type not in ("image", "text", "table", "chart", "audio"):
        return None

    raw_filename = hit.get("source") or hit.get("path") or ""
    document_name = os.path.basename(str(raw_filename)) if raw_filename else ""

    try:
        page_number = int(hit.get("page_number") or 0)
    except (TypeError, ValueError):
        page_number = 0

    score = float(
        hit.get("_score") or hit.get("relevance_score") or hit.get("_distance") or 0.0
    )

    return SourceResult(
        content=content,
        document_type=document_type,
        document_name=document_name,
        score=score,
        stage=stage,
        metadata=SourceMetadata(
            page_number=page_number,
            description=hit.get("text") or "",
            content_metadata=hit.get("metadata") or {},
        ),
    )


async def run_nrl_native_search(
    *,
    config: NvidiaRAGConfig,
    query: str,
    collection_names: list[str],
    vdb_endpoint: str | None,
    vdb_top_k: int,
    reranker_top_k: int,
    enable_reranker: bool,
    reranker_model: str | None,
    reranker_endpoint: str | None,
    confidence_threshold: float | None,
    stage: str,
) -> Citations:
    """Run NRL-native retrieval: one Retriever per collection, queried
    concurrently, results merged/sorted/filtered into a Citations object with
    the identical shape NvidiaRAG.search() returns.
    """
    effective_top_k = vdb_top_k if enable_reranker else reranker_top_k

    def _query_one(collection_name: str) -> list[dict[str, Any]]:
        try:
            retriever = build_nrl_retriever(
                config=config,
                collection_name=collection_name,
                vdb_endpoint=vdb_endpoint,
                top_k=effective_top_k,
                enable_reranker=enable_reranker,
                reranker_model=reranker_model,
                reranker_endpoint=reranker_endpoint,
            )
            # Retriever.query() is SYNCHRONOUS — this runs via asyncio.to_thread.
            return retriever.query(query, top_k=effective_top_k)
        except Exception:
            logger.exception(
                "[NRL Native Retriever] Query failed for collection %r",
                collection_name,
            )
            return []

    hit_lists = await asyncio.gather(
        *(asyncio.to_thread(_query_one, name) for name in collection_names)
    )

    all_results: list[SourceResult] = []
    for hits in hit_lists:
        for hit in hits:
            sr = _hit_to_source_result(hit, stage)
            if sr is not None:
                all_results.append(sr)

    # Merge all collections' results, then apply a single global sort + top-k
    # truncation — mirrors search()'s multi-collection merge-then-rerank
    # pattern (main.py:1480-1533).
    all_results.sort(key=lambda r: r.score, reverse=True)

    if enable_reranker:
        all_results = all_results[:reranker_top_k]
        if confidence_threshold is not None and confidence_threshold > 0.0:
            all_results = [r for r in all_results if r.score >= confidence_threshold]
    else:
        all_results = all_results[:vdb_top_k]

    return Citations(total_results=len(all_results), results=all_results)
