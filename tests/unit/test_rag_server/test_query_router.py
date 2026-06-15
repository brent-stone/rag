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

"""Unit tests for the agentic RAG per-query reasoning router.

No network: the classifier is a pure rules pass.
"""

from __future__ import annotations

import pytest

from nvidia_rag.rag_server.agentic_rag.query_router import (
    COMPLEX,
    SIMPLE,
    VALID_ROLES,
    build_thinking_overrides,
    classify_query,
    parse_routed_roles,
)


class TestClassifyQueryComplex:
    """Computation / comparison / multi-hop questions -> complex (thinking-ON)."""

    @pytest.mark.parametrize(
        "query",
        [
            "What was the revenue growth from 2019 to 2020?",
            "Calculate the total operating expenses for FY2021.",
            "How much did net income increase year-over-year?",
            "Compare the gross margins of Apple and Microsoft.",
            "What is the difference between the 2020 and 2021 cash flow?",
            "Which company had a higher P/E ratio than its peers?",
            "What percentage of revenue came from international sales?",
            "Sum the quarterly dividends paid in 2022.",
            "What is 12 * 8 plus the closing balance?",
            "Show the revenue in $ and the change versus last year.",
            "List the CEOs of both companies respectively.",
            "What is the average headcount across each of the divisions?",
        ],
    )
    def test_complex_queries(self, query: str) -> None:
        decision = classify_query(query)
        assert decision.label == COMPLEX
        assert decision.thinking is True
        assert decision.uncertain is False
        assert decision.matched is not None

    def test_multiple_questions_is_complex(self) -> None:
        decision = classify_query("Who founded it? When was it founded?")
        assert decision.label == COMPLEX
        assert decision.thinking is True
        assert decision.matched == "multiple '?'"


class TestClassifyQuerySimple:
    """High-confidence lookups / summaries with no complex signal -> simple."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is the capital of France?",
            "Who is the current CEO of the company?",
            "Where are the headquarters located?",
            "Define amortization.",
            "Describe the company's main product line.",
            "Summarize the introduction section.",
            "Explain the dividend policy.",
            "Tell me about the founding history.",
            "Give me an overview of the report.",
            "List the board members.",
        ],
    )
    def test_simple_queries(self, query: str) -> None:
        decision = classify_query(query)
        assert decision.label == SIMPLE
        assert decision.thinking is False
        assert decision.uncertain is False


class TestClassifyQuerySafetyBias:
    """Uncertain / empty queries fall back to complex (thinking-ON)."""

    def test_empty_query_is_complex_uncertain(self) -> None:
        for q in ("", "   ", None):  # type: ignore[arg-type]
            decision = classify_query(q)  # type: ignore[arg-type]
            assert decision.label == COMPLEX
            assert decision.thinking is True
            assert decision.uncertain is True

    def test_unrecognized_query_defaults_to_complex(self) -> None:
        # No simple opener, no complex signal -> safety default.
        decision = classify_query("Revenue figures for the fiscal period.")
        assert decision.label == COMPLEX
        assert decision.thinking is True
        assert decision.uncertain is True

    def test_complex_signal_beats_simple_opener(self) -> None:
        # "What is" opener but a complex signal present -> complex wins.
        decision = classify_query("What is the total revenue for 2021?")
        assert decision.label == COMPLEX
        assert decision.thinking is True


class TestParseRoutedRoles:
    def test_parses_subset_in_canonical_order(self) -> None:
        assert parse_routed_roles("synthesis,task") == ["task", "synthesis"]

    def test_drops_unknown_roles(self) -> None:
        assert parse_routed_roles("task,bogus,synthesis") == ["task", "synthesis"]

    def test_empty_falls_back_to_all_roles(self) -> None:
        assert parse_routed_roles("") == list(VALID_ROLES)
        assert parse_routed_roles("   ") == list(VALID_ROLES)

    def test_all_invalid_falls_back_to_all_roles(self) -> None:
        assert parse_routed_roles("nope,nada") == list(VALID_ROLES)

    def test_case_and_whitespace_insensitive(self) -> None:
        assert parse_routed_roles(" Task , SYNTHESIS ") == ["task", "synthesis"]


class TestBuildThinkingOverrides:
    def test_complex_turns_all_routed_roles_on(self) -> None:
        decision = classify_query("Calculate the total.")
        overrides = build_thinking_overrides(decision, list(VALID_ROLES))
        assert overrides == dict.fromkeys(VALID_ROLES, True)

    def test_simple_turns_all_routed_roles_off(self) -> None:
        decision = classify_query("What is the capital of France?")
        overrides = build_thinking_overrides(decision, list(VALID_ROLES))
        assert overrides == dict.fromkeys(VALID_ROLES, False)

    def test_only_routed_roles_present(self) -> None:
        decision = classify_query("What is the capital of France?")
        overrides = build_thinking_overrides(decision, ["task", "synthesis"])
        assert set(overrides) == {"task", "synthesis"}
        assert overrides == {"task": False, "synthesis": False}
