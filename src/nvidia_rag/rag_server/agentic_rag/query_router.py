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

"""Per-query reasoning router for agentic RAG.

A lightweight, network-free classifier that labels each incoming question so
the agentic pipeline can pick a reasoning ("thinking") mode for *that request
only*:

* ``simple``  — fact lookup, summarise/describe, single-doc answer. These are
  cheap to answer correctly without chain-of-thought, so the router turns
  thinking OFF (fast / cheap path).
* ``complex`` — numeric computation, comparison/arithmetic, multi-hop chaining.
  These need step-by-step reasoning, so the router turns thinking ON.

Design notes
------------
* **Cheap.** The classifier is a single regex pass over the query string — no
  LLM call, no network, no measurable per-query latency.
* **Bias toward safety.** A hard question mislabelled ``simple`` and sent down
  the fast path produces a *wrong* answer, which is the exact failure we must
  avoid. So the rules over-match ``complex``: ``simple`` is only returned on a
  high-confidence lookup/summary pattern with *no* complex signal present, and
  anything ambiguous (or empty) falls back to ``complex`` (thinking-ON).

The label is mapped to a per-role ``enable_thinking`` override via
:func:`build_thinking_overrides`, applied through the request-scoped
``_agentic_llm_overrides`` ContextVar (see ``builder.AgenticLLMOverrides``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Valid agentic role names the router may control. Mirrors the per-role LLM
# configs in agentic_rag_config.py / the role properties in agentic_rag.py.
VALID_ROLES: tuple[str, ...] = ("planner", "task", "seed_gen", "synthesis")

SIMPLE = "simple"
COMPLEX = "complex"


# ---------------------------------------------------------------------------
# Signal patterns
#
# COMPLEX patterns are intentionally broad — over-matching complex is the safe
# error direction (worst case: a simple query runs slightly slower). SIMPLE
# patterns are narrow, high-confidence lookup/summary openers.
# ---------------------------------------------------------------------------

_COMPLEX_PATTERNS: tuple[str, ...] = (
    # --- arithmetic / computation -----------------------------------------
    r"\bcalculat(?:e|ed|es|ing|ion)\b",
    r"\bcomput(?:e|ed|es|ing|ation)\b",
    r"\bhow much\b",
    r"\bhow many\b",
    r"\b(?:sum|total|subtotal|aggregate|average|mean|median|ratio|"
    r"proportion|percentage|percent)\b",
    r"\b(?:difference|differ|increase|increased|decrease|decreased|decline|"
    r"declined|growth|grew|grow|reduction|reduced)\b",
    r"\b(?:rate|margin|cagr|yoy|year[- ]over[- ]year|year[- ]on[- ]year|"
    r"per[- ]annum|per[- ]share|per[- ]capita)\b",
    r"\b(?:more|less|fewer|greater|higher|lower|larger|smaller)\s+than\b",
    # --- comparison -------------------------------------------------------
    r"\bcompare[ds]?\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bbetween\b.+\band\b",
    # --- symbols / explicit numeric operators -----------------------------
    r"%",
    r"[$€£¥]",
    r"\d\s*[-+*/x×÷]\s*\d",
    # --- multi-hop / chaining cues ----------------------------------------
    r"\band then\b",
    r"\bafter (?:that|which)\b",
    r"\bbased on\b",
    r"\bboth\b",
    r"\beach of\b",
    r"\brespectively\b",
    r"\bcombined\b",
    r"\bin total\b",
)

_SIMPLE_PATTERNS: tuple[str, ...] = (
    # High-confidence single-fact lookups: "what is", "who was", "where are"…
    r"^\s*(?:what|who|where|when|which)\s+(?:is|are|was|were)\b",
    # Open-ended summary / description openers.
    r"^\s*(?:define|describe|summari[sz]e|explain|list|name|outline)\b",
    r"^\s*tell me about\b",
    r"^\s*(?:give|provide)\s+(?:me\s+)?an?\s+(?:overview|summary)\b",
    r"\bdefinition of\b",
)

_COMPLEX_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS
)
_SIMPLE_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS
)


@dataclass(frozen=True)
class RoutingDecision:
    """Outcome of classifying one query.

    Attributes:
        label:     ``"simple"`` or ``"complex"``.
        thinking:  True when reasoning should be ON (i.e. label == complex).
        uncertain: True when no clear signal matched and the safety default
                   (complex) was applied.
        reason:    Short human-readable explanation for logs.
        matched:   The first signal pattern that matched, if any.
    """

    label: str
    thinking: bool
    uncertain: bool
    reason: str
    matched: str | None = None


def classify_query(query: str) -> RoutingDecision:
    """Classify ``query`` as ``simple`` or ``complex`` using a cheap rules pass.

    Resolution order (first match wins):
      1. Any COMPLEX signal  -> complex (thinking-ON).
      2. Else a SIMPLE signal -> simple (thinking-OFF).
      3. Otherwise (no signal / empty) -> complex (thinking-ON, safety default).

    Multiple distinct questions in one prompt (``>= 2`` ``?``) are treated as
    complex (multi-hop).

    Args:
        query: The user question (already query-rewritten if applicable).

    Returns:
        A :class:`RoutingDecision`.
    """
    text = (query or "").strip()
    if not text:
        return RoutingDecision(
            label=COMPLEX,
            thinking=True,
            uncertain=True,
            reason="empty query -> safety default (thinking-ON)",
        )

    if text.count("?") >= 2:
        return RoutingDecision(
            label=COMPLEX,
            thinking=True,
            uncertain=False,
            reason="multiple questions -> complex (multi-hop)",
            matched="multiple '?'",
        )

    for pat in _COMPLEX_RE:
        if pat.search(text):
            return RoutingDecision(
                label=COMPLEX,
                thinking=True,
                uncertain=False,
                reason="matched complex signal -> thinking-ON",
                matched=pat.pattern,
            )

    for pat in _SIMPLE_RE:
        if pat.search(text):
            return RoutingDecision(
                label=SIMPLE,
                thinking=False,
                uncertain=False,
                reason="matched simple signal, no complex signal -> thinking-OFF",
                matched=pat.pattern,
            )

    return RoutingDecision(
        label=COMPLEX,
        thinking=True,
        uncertain=True,
        reason="no clear signal -> safety default (thinking-ON)",
    )


def parse_routed_roles(raw: str) -> list[str]:
    """Parse the ``AGENTIC_QUERY_ROUTING_ROLES`` comma string into role names.

    Unknown role names are dropped. If nothing valid remains (empty / all
    invalid), falls back to all four roles so a misconfiguration never silently
    disables routing for every role.

    Args:
        raw: Comma-separated role names, e.g. ``"task,synthesis"``.

    Returns:
        Ordered list of valid role names (subset of :data:`VALID_ROLES`).
    """
    requested = [r.strip().lower() for r in (raw or "").split(",") if r.strip()]
    valid = [r for r in VALID_ROLES if r in requested]
    return valid or list(VALID_ROLES)


def build_thinking_overrides(
    decision: RoutingDecision, routed_roles: list[str]
) -> dict[str, bool]:
    """Map a routing decision to a per-role ``enable_thinking`` override dict.

    Only ``routed_roles`` are included; roles omitted from the routing config
    keep their server-wide ``AGENTIC_*_LLM_ENABLE_THINKING`` default (because
    they are absent from the returned dict, the resolver falls through to
    config).

    Args:
        decision:     Result of :func:`classify_query`.
        routed_roles: Roles the router is allowed to control.

    Returns:
        ``{role: thinking_bool}`` for each routed role.
    """
    return dict.fromkeys(routed_roles, decision.thinking)
