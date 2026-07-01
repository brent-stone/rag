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

"""Unit tests for the NRL-native agentic RAG retrieval path (no network calls)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvidia_rag.rag_server.agentic_rag import nrl_native_retriever as nrl_mod
from nvidia_rag.rag_server.agentic_rag.nrl_native_retriever import (
    _hit_to_source_result,
    build_nrl_retriever,
    run_nrl_native_search,
)
from nvidia_rag.rag_server.response_generator import Citations


def _fake_nemo_retriever_module(retriever_cls):
    """Install a stub `nemo_retriever.retriever` module so
    `build_nrl_retriever`'s deferred `from nemo_retriever.retriever
    import Retriever` resolves without the optional dependency installed.
    """
    pkg = types.ModuleType("nemo_retriever")
    retriever_mod = types.ModuleType("nemo_retriever.retriever")
    retriever_mod.Retriever = retriever_cls
    sys.modules["nemo_retriever"] = pkg
    sys.modules["nemo_retriever.retriever"] = retriever_mod


@pytest.fixture(autouse=True)
def _cleanup_fake_nemo_retriever():
    yield
    for name in ("nemo_retriever", "nemo_retriever.retriever"):
        sys.modules.pop(name, None)


def _make_config(*, reranker_endpoint: str = "http://rerank", vector_store_url: str = "/lancedb"):
    config = MagicMock()
    config.embeddings.model_name = "embed-model"
    config.embeddings.server_url = "http://embed"
    config.embeddings.get_api_key.return_value = "embed-key"
    config.ranking.model_name = "rerank-model"
    config.ranking.server_url = reranker_endpoint
    config.ranking.get_api_key.return_value = "rerank-key"
    config.vector_store.url = vector_store_url
    return config


class TestBuildNrlRetriever:
    def test_constructs_expected_kwargs_with_reranker_enabled(self) -> None:
        retriever_cls = MagicMock()
        _fake_nemo_retriever_module(retriever_cls)
        config = _make_config()

        build_nrl_retriever(
            config=config,
            collection_name="my_collection",
            vdb_endpoint=None,
            top_k=10,
            enable_reranker=True,
            reranker_model=None,
            reranker_endpoint=None,
        )

        retriever_cls.assert_called_once()
        kwargs = retriever_cls.call_args.kwargs
        assert kwargs["run_mode"] == "service"
        assert kwargs["top_k"] == 10
        assert kwargs["rerank"] is True
        assert kwargs["vdb_kwargs"] == {"uri": "/lancedb", "table_name": "my_collection"}
        assert kwargs["embed_kwargs"] == {
            "model_name": "embed-model",
            "embed_invoke_url": "http://embed",
            "api_key": "embed-key",
        }
        assert kwargs["rerank_kwargs"] == {
            "model_name": "rerank-model",
            "rerank_invoke_url": "http://rerank",
            "api_key": "rerank-key",
        }

    def test_rerank_kwargs_empty_when_reranker_disabled(self) -> None:
        retriever_cls = MagicMock()
        _fake_nemo_retriever_module(retriever_cls)
        config = _make_config()

        build_nrl_retriever(
            config=config,
            collection_name="c",
            vdb_endpoint=None,
            top_k=5,
            enable_reranker=False,
            reranker_model=None,
            reranker_endpoint=None,
        )

        kwargs = retriever_cls.call_args.kwargs
        assert kwargs["rerank"] is False
        assert kwargs["rerank_kwargs"] == {}

    def test_vdb_endpoint_override_takes_precedence_over_config_url(self) -> None:
        retriever_cls = MagicMock()
        _fake_nemo_retriever_module(retriever_cls)
        config = _make_config(vector_store_url="/default/lancedb")

        build_nrl_retriever(
            config=config,
            collection_name="c",
            vdb_endpoint="/override/lancedb",
            top_k=5,
            enable_reranker=False,
            reranker_model=None,
            reranker_endpoint=None,
        )

        kwargs = retriever_cls.call_args.kwargs
        assert kwargs["vdb_kwargs"] == {"uri": "/override/lancedb", "table_name": "c"}

    def test_new_instance_constructed_on_every_call(self) -> None:
        retriever_cls = MagicMock()
        _fake_nemo_retriever_module(retriever_cls)
        config = _make_config()

        build_nrl_retriever(
            config=config,
            collection_name="c",
            vdb_endpoint=None,
            top_k=5,
            enable_reranker=False,
            reranker_model=None,
            reranker_endpoint=None,
        )
        build_nrl_retriever(
            config=config,
            collection_name="c",
            vdb_endpoint=None,
            top_k=5,
            enable_reranker=False,
            reranker_model=None,
            reranker_endpoint=None,
        )

        assert retriever_cls.call_count == 2


class TestHitToSourceResult:
    def test_text_hit_maps_content_directly(self) -> None:
        hit = {"text": "hello world", "content_type": "text", "_score": 0.9}
        result = _hit_to_source_result(hit, stage="rag")
        assert result is not None
        assert result.document_type == "text"
        assert result.content == "hello world"
        assert result.score == 0.9
        assert result.stage == "rag"

    def test_image_hit_fetches_object_store_and_base64_encodes(self) -> None:
        hit = {
            "stored_image_uri": "s3://bucket/img.png",
            "content_type": "image",
            "text": "a chart",
            "_score": 0.5,
        }
        fake_operator = Mock()
        fake_operator.get_object_from_uri.return_value = b"rawbytes"
        with patch.object(
            nrl_mod, "get_object_store_operator_instance", return_value=fake_operator
        ):
            result = _hit_to_source_result(hit, stage="rag")
        assert result is not None
        assert result.document_type == "image"
        import base64

        assert result.content == base64.b64encode(b"rawbytes").decode("ascii")

    def test_skips_hit_with_no_renderable_content(self) -> None:
        hit = {"text": "", "content_type": "text"}
        assert _hit_to_source_result(hit, stage="rag") is None

    def test_skips_hit_when_object_store_fetch_fails(self) -> None:
        hit = {"stored_image_uri": "s3://bucket/img.png", "content_type": "image"}
        fake_operator = Mock()
        fake_operator.get_object_from_uri.side_effect = Exception("boom")
        with patch.object(
            nrl_mod, "get_object_store_operator_instance", return_value=fake_operator
        ):
            assert _hit_to_source_result(hit, stage="rag") is None

    def test_unmapped_content_type_defaults_to_text_or_image(self) -> None:
        hit = {"text": "content", "content_type": "unknown_type"}
        result = _hit_to_source_result(hit, stage="rag")
        assert result is not None
        assert result.document_type == "text"

    def test_score_falls_back_through_score_relevance_distance(self) -> None:
        hit = {"text": "x", "_distance": 0.25}
        result = _hit_to_source_result(hit, stage="rag")
        assert result is not None
        assert result.score == 0.25

    def test_document_name_from_source_basename(self) -> None:
        hit = {"text": "x", "source": "/path/to/doc.pdf"}
        result = _hit_to_source_result(hit, stage="rag")
        assert result is not None
        assert result.document_name == "doc.pdf"

    def test_page_number_defaults_to_zero_on_bad_value(self) -> None:
        hit = {"text": "x", "page_number": "not-a-number"}
        result = _hit_to_source_result(hit, stage="rag")
        assert result is not None
        assert result.metadata.page_number == 0


class TestRunNrlNativeSearch:
    @pytest.mark.asyncio
    async def test_single_collection_happy_path(self) -> None:
        fake_retriever = Mock()
        fake_retriever.query.return_value = [
            {"text": "a", "_score": 0.4},
            {"text": "b", "_score": 0.9},
        ]
        with patch.object(nrl_mod, "build_nrl_retriever", return_value=fake_retriever):
            citations = await run_nrl_native_search(
                config=MagicMock(),
                query="q",
                collection_names=["col_a"],
                vdb_endpoint=None,
                vdb_top_k=10,
                reranker_top_k=5,
                enable_reranker=True,
                reranker_model=None,
                reranker_endpoint=None,
                confidence_threshold=None,
                stage="rag",
            )
        assert isinstance(citations, Citations)
        assert citations.total_results == 2
        # Sorted descending by score.
        assert citations.results[0].content == "b"
        assert citations.results[1].content == "a"

    @pytest.mark.asyncio
    async def test_multi_collection_merges_and_sorts_globally(self) -> None:
        def fake_build(config, collection_name, **kwargs):
            retriever = Mock()
            if collection_name == "col_a":
                retriever.query.return_value = [{"text": "low", "_score": 0.1}]
            else:
                retriever.query.return_value = [{"text": "high", "_score": 0.99}]
            return retriever

        with patch.object(nrl_mod, "build_nrl_retriever", side_effect=fake_build):
            citations = await run_nrl_native_search(
                config=MagicMock(),
                query="q",
                collection_names=["col_a", "col_b"],
                vdb_endpoint=None,
                vdb_top_k=10,
                reranker_top_k=10,
                enable_reranker=True,
                reranker_model=None,
                reranker_endpoint=None,
                confidence_threshold=None,
                stage="rag",
            )
        assert [r.content for r in citations.results] == ["high", "low"]

    @pytest.mark.asyncio
    async def test_confidence_threshold_filters_only_when_reranker_enabled(self) -> None:
        fake_retriever = Mock()
        fake_retriever.query.return_value = [
            {"text": "keep", "_score": 0.9},
            {"text": "drop", "_score": 0.1},
        ]
        with patch.object(nrl_mod, "build_nrl_retriever", return_value=fake_retriever):
            citations = await run_nrl_native_search(
                config=MagicMock(),
                query="q",
                collection_names=["col_a"],
                vdb_endpoint=None,
                vdb_top_k=10,
                reranker_top_k=10,
                enable_reranker=True,
                reranker_model=None,
                reranker_endpoint=None,
                confidence_threshold=0.5,
                stage="rag",
            )
        assert [r.content for r in citations.results] == ["keep"]

    @pytest.mark.asyncio
    async def test_confidence_threshold_ignored_when_reranker_disabled(self) -> None:
        fake_retriever = Mock()
        fake_retriever.query.return_value = [
            {"text": "keep", "_score": 0.9},
            {"text": "also_kept", "_score": 0.1},
        ]
        with patch.object(nrl_mod, "build_nrl_retriever", return_value=fake_retriever):
            citations = await run_nrl_native_search(
                config=MagicMock(),
                query="q",
                collection_names=["col_a"],
                vdb_endpoint=None,
                vdb_top_k=10,
                reranker_top_k=10,
                enable_reranker=False,
                reranker_model=None,
                reranker_endpoint=None,
                confidence_threshold=0.5,
                stage="rag",
            )
        assert citations.total_results == 2

    @pytest.mark.asyncio
    async def test_reranker_top_k_truncation(self) -> None:
        fake_retriever = Mock()
        fake_retriever.query.return_value = [
            {"text": f"hit{i}", "_score": 1.0 - i * 0.01} for i in range(5)
        ]
        with patch.object(nrl_mod, "build_nrl_retriever", return_value=fake_retriever):
            citations = await run_nrl_native_search(
                config=MagicMock(),
                query="q",
                collection_names=["col_a"],
                vdb_endpoint=None,
                vdb_top_k=10,
                reranker_top_k=2,
                enable_reranker=True,
                reranker_model=None,
                reranker_endpoint=None,
                confidence_threshold=None,
                stage="rag",
            )
        assert citations.total_results == 2

    @pytest.mark.asyncio
    async def test_per_collection_failure_isolated(self) -> None:
        def fake_build(config, collection_name, **kwargs):
            retriever = Mock()
            if collection_name == "bad_col":
                retriever.query.side_effect = RuntimeError("boom")
            else:
                retriever.query.return_value = [{"text": "ok", "_score": 0.5}]
            return retriever

        with patch.object(nrl_mod, "build_nrl_retriever", side_effect=fake_build):
            citations = await run_nrl_native_search(
                config=MagicMock(),
                query="q",
                collection_names=["bad_col", "good_col"],
                vdb_endpoint=None,
                vdb_top_k=10,
                reranker_top_k=10,
                enable_reranker=True,
                reranker_model=None,
                reranker_endpoint=None,
                confidence_threshold=None,
                stage="rag",
            )
        assert citations.total_results == 1
        assert citations.results[0].content == "ok"
