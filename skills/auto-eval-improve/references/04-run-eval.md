# Stage 4 — Run ingestion + evaluation (in the background)

Goal: run `evaluate_rag.py` for the chosen datasets, in agentic (or standard) mode, as a long-running
**background** process. It does both ingestion and evaluation in one pass.

## 4.0 Create the experiment folder (do this first, once per invocation)

Every artifact for this campaign — logs, result snapshots, intermediate state, the final report — lives
under a single **experiment folder** inside `$SCRIPTS_DIR/experiments/`. One experiment = one invocation
of this skill (baseline + up to 3 improve cycles, across all datasets you run this turn). Create it once,
before launching any eval, and reuse `$EXP_DIR` for every later stage.

```bash
cd "$SCRIPTS_DIR"
MODE=agentic                                    # agentic | standard (the mode confirmed in 4.1)
EXP_TS=$(date +%Y%m%d_%H%M%S)
EXP_DIR="experiments/exp_${EXP_TS}_${MODE}"
mkdir -p "$EXP_DIR"
echo "Experiment: $EXP_DIR"
```

Seed `state.md` — the experiment's **manifest + live progress checkpoint** (replaces the old ad-hoc
`AUTO_EVAL_STATE.md`). Fill the header with the real, detected values (mode, datasets, deployment
endpoints / model / key `AGENTIC_*` env from the active env file). Stage 5.5 appends one row per cycle;
treat it as the resume/checkpoint doc:

```markdown
# Experiment exp_<ts>_<mode>

- **Mode:** agentic | standard
- **Datasets:** financebench, kg_rag, …
- **rag-server / ingestor:** localhost:8081 / http://localhost:8082
- **vdb / llm endpoint / model:** http://elasticsearch:9200 / nim-llm:8000 / nvidia/nemotron-3-super-120b-a12b
- **Key env:** ENABLE_AGENTIC_RAG=true, AGENTIC_VERIFICATION_ENABLED=false, …
- **Started:** <ts>

## Progress
| Dataset | Cycle | Snapshot | Change applied | e2e | ctx_rel | grounded | recall@1 | p50_latency_s | ttft_s | Verdict |
|---------|-------|----------|----------------|-----|---------|----------|----------|---------------|--------|---------|

Record **latency alongside accuracy** every cycle (p50 end-to-end + mean/p50 TTFT, from the `latency` block
in `rag_<dataset>_evaluation_metrics.json`). The verdict must weigh both: an accuracy-neutral change that
cuts p50 latency is a *win*; an accuracy gain that costs latency is a *trade-off* to call out explicitly.

## Observations (fill in during Stage 6.3 — the evidence base for cross-dataset recommendations)

Record findings **per dataset** so code-change recommendations can be judged across the WHOLE experiment,
not from one dataset in isolation. A code/flow change affects every dataset, so it must be justified by
evidence spanning all of them (see reference 06, 6.4 and D.2). Keep adding rows as you analyze each run.

### Per-dataset failure analysis
| Dataset | n queries | n hard-fails (e2e<0.5) | Dominant failure mode(s) | Example failing indices | Notes |
|---------|-----------|------------------------|--------------------------|-------------------------|-------|

### Cross-dataset failure-mode rollup (which modes recur across datasets)
| Failure mode | Datasets affected | Approx. share of fails | Where in pipeline (retrieval / planner / task-answer / synthesis / eval) |
|--------------|-------------------|------------------------|--------------------------------------------------------------------------|

### Candidate code-change hypotheses (each must be assessed against ALL datasets before recommending)
| Hypothesis (what + where in `rag/`) | Targets which failure mode | Datasets it should HELP | Datasets it could HURT / risk | Cross-dataset verdict |
|-------------------------------------|----------------------------|-------------------------|-------------------------------|-----------------------|
```

The resulting tree (filled in by Stages 4–6):

```
experiments/exp_<ts>_<mode>/
├── state.md                       # manifest + live cycle log (this file)
├── report.md                      # final consolidated report (Stage 6.8)
└── <dataset>/                     # financebench/, kg_rag/, …
    ├── baseline_<TS>/             # baseline | cycle1 | cycle2 | cycle3
    │   ├── rag_<dataset>_evaluation_{metrics,results,data,summary}.json
    │   ├── rag_<dataset>_stream_tokens.json   # per-stage streamed-output token breakdown (Stage 5.3 copies it too)
    │   ├── rag_<dataset>_token_usage.json     # wire prompt/completion/total per query (Stage 5.3 copies it too)
    │   ├── failure.txt            # only if >50% queries failed
    │   ├── eval.log               # this run's tee'd log
    │   └── REPRODUCE.md           # self-contained rerun steps (Stage 5.4)
    └── cycle1_<TS>/ …
```

> `--output_dir` stays at its default `results` — the script writes `results/<dataset>/` as **transient
> scratch**, and each stage copies that snapshot into `$EXP_DIR/<dataset>/<cycle>_<TS>/` (Stage 5.3). Do
> not point `--output_dir` at the experiment folder (the script would append a second `/<dataset>`).

## 4.1 Ask which datasets to run (QA)

Use `AskUserQuestion` (multi-select) to let the user pick. Default selection if they have no preference:
**`google_frames`, `financebench`, `kg_rag`, `hotpotqa`**. **Launch one independent background process per
dataset** (not one command with a space-separated `--datasets` list): a single failure then doesn't block
the rest, each is timestamped independently (Stage 5), and the per-dataset processes can run **in parallel**
on one rag-server (§4.4). Confirm only datasets already downloaded in Stage 3.

Also confirm the **mode**: agentic (default) or standard. This decides whether `--agentic` is on the
command (and whether `ENABLE_AGENTIC_RAG=true` was set in Stage 1).

## 4.2 Read the argparse — build the command from the script, not the example

Before constructing anything:

```bash
cd "$SCRIPTS_DIR" && python3 evaluate_rag.py --help
```

Key flags (from `evaluate_rag.py`):

| Flag | Meaning / value for this skill |
|------|--------------------------------|
| `--datasets` | **Plural**, space-separated list (e.g. `--datasets financebench kg_rag`). Names must be in `ALLOWED_DATASETS`. |
| `--host` / `--port` | rag-server reachable from where the script runs — usually `localhost` / `8081`. **Client-side.** |
| `--ingestor_server_url` | rag ingestor base URL, **no `/v1` suffix** (code appends it) — usually `http://localhost:8082`. **Client-side.** |
| `--rag_api_version` | `2` for the current RAG (v2 schema, metrics file, citations). Use `2`. |
| `--vdb_endpoint` | Vector DB endpoint **as the servers see it** (Docker-internal). Elasticsearch: `http://elasticsearch:9200`; Milvus: `http://milvus:19530`. **Server-side.** |
| `--llm_endpoint` | LLM endpoint **as the rag-server sees it**. Accepts a **full URL** for cloud (e.g. `https://inference-api.nvidia.com/v1`) **or** a `host:port` for on-prem (e.g. `nim-llm:8000`). **Always pass it**, matching the deployment's `APP_LLM_SERVERURL`. **Server-side.** |
| `--model` | LLM model name. **Always pass it**, matching the deployment's `APP_LLM_MODELNAME` (e.g. `nvidia/nvidia/nemotron-3-super-v3`). **Do NOT omit it** — there is no "use server default": argparse defaults `--model` to `nvdev/meta/llama-3.3-70b-instruct` and sends that in every request, silently overriding the deployed model. |
| `--agentic` | **Bare flag.** Present ⇒ agentic run (`agentic:true` per request). Omit for standard RAG. |
| `--top_k` | Reranker top-k for generation (e.g. `10`). |
| `--output_dir` | Default `results`. Relative to CWD ⇒ lands in `scripts/results/`. |
| `--thread` | Concurrency for ingestion + generation (e.g. `16`–`32`). |
| `--embedding_dimension` | Match the embedding model (default `2048`). |
| `--force_ingestion` | Deletes + re-ingests the collection. **Destructive** — confirm with the user. |
| `--skip_ingestion` / `--skip_evaluation` | Reuse an already-ingested collection / ingest only. |
| `--skip_inference` / `--data_input_json` / `--judge_model` | **Re-evaluation only.** `--skip_inference` skips ingestion+inference and loads a pre-generated `rag_<dataset>_evaluation_data.json` given by `--data_input_json`, then runs only the RAGAS judge. `--judge_model` overrides the judge LLM (e.g. a second judge "B"); defaults to the script's `JUDGE_MODEL`. `--skip_inference` requires `--data_input_json` and **exactly one** `--datasets` value, and implies `--skip_ingestion`. Use a **distinct `--output_dir`** so re-eval results don't overwrite the original judge's results. |
| `--chunk_size` / `--chunk_overlap` / `--batch_size` | Ingestion params; defaults usually fine. |

> **Endpoint gotcha (important):** `--host/--port` and `--ingestor_server_url` are called *by the eval
> client*, so use `localhost` when the script runs on the host. `--vdb_endpoint` and `--llm_endpoint` are
> forwarded into the request and resolved *by the rag-server* — so `--vdb_endpoint` uses the **container
> hostname** (`elasticsearch`, `milvus`), while `--llm_endpoint` is whatever the rag-server uses to reach
> the LLM: a **container `host:port`** for on-prem NIM (`nim-llm:8000`) or a **full external URL** for
> cloud (`https://inference-api.nvidia.com/v1`). Read the actual values from the active env file
> (`APP_VECTORSTORE_URL`, `APP_LLM_SERVERURL`, `APP_LLM_MODELNAME`) rather than guessing; mismatches cause
> empty contexts or connection errors.

## 4.3 Example command (agentic)

Adapt to what Stage 1 detected. Agentic financebench run on a host with Elasticsearch + on-prem LLM:

```bash
cd "$SCRIPTS_DIR"
source .venv/bin/activate
export NVIDIA_API_KEY="$NVIDIA_API_KEY"

PYTHONUNBUFFERED=1 python3 evaluate_rag.py \
  --datasets financebench \
  --host localhost --port 8081 \
  --ingestor_server_url http://localhost:8082 \
  --rag_api_version 2 \
  --vdb_endpoint http://elasticsearch:9200 \
  --llm_endpoint nim-llm:8000 \
  --model nvidia/nemotron-3-super-120b-a12b \
  --top_k 10 \
  --thread 16 \
  --output_dir results \
  --agentic
```

**Always pass `--model` and `--llm_endpoint`** matching the deployment's `APP_LLM_MODELNAME` /
`APP_LLM_SERVERURL` — never rely on omission (see the flag table). `--llm_endpoint` takes a full URL for
cloud or a `host:port` for on-prem:

```bash
# NVIDIA-hosted / cloud LLM (full URL):
python3 evaluate_rag.py --port 8081 --host localhost --datasets kg_rag --top_k 10 --rag_api_version 2 \
  --vdb_endpoint http://elasticsearch:9200 \
  --llm_endpoint https://inference-api.nvidia.com/v1 --model nvidia/nvidia/nemotron-3-super-v3

# On-prem NIM LLM (host:port):
python3 evaluate_rag.py --port 8081 --host localhost --datasets kg_rag --top_k 10 --rag_api_version 2 \
  --vdb_endpoint http://elasticsearch:9200 \
  --llm_endpoint nim-llm:8000 --model nvidia/nvidia/nemotron-3-super-v3
```

For **standard RAG**: drop `--agentic` (and ensure `ENABLE_AGENTIC_RAG` is not forcing agentic on the
server). Everything else — including passing `--model` and `--llm_endpoint` — is identical.

## 4.3a Re-evaluate an existing run with a different judge model

To re-score an already-generated `data.json` with a different judge — **without** re-running inference —
pass `--skip_inference` + `--data_input_json` (and optionally `--judge_model`). This skips ingestion and
generation entirely and runs only the RAGAS judge, so it is fast and needs only `NVIDIA_API_KEY`.

```bash
cd "$SCRIPTS_DIR"
source .venv/bin/activate
export NVIDIA_API_KEY="$NVIDIA_API_KEY"

PYTHONUNBUFFERED=1 python3 evaluate_rag.py \
  --datasets financebench \
  --host localhost --port 8081 \
  --rag_api_version 2 \
  --skip_inference \
  --data_input_json results/financebench/rag_financebench_evaluation_data.json \
  --judge_model nvidia/openai/gpt-oss-120b \
  --top_k 10 \
  --output_dir results_judgeB
```

> **Notes:** `--skip_inference` requires exactly one `--datasets` value and an existing
> `--data_input_json` file (it errors otherwise), and it implies `--skip_ingestion`. `--host`/`--port`
> stay required by argparse but are **unused** on this path — any reachable/placeholder value is fine.
> Point `--output_dir` at a **fresh directory** (e.g. `results_judgeB`) so the re-eval's
> `rag_<dataset>_evaluation_results.json` / `_summary.json` / `_metrics.json` don't overwrite the
> original judge's outputs. Omit `--judge_model` to re-run with the default judge.

## 4.4 Launch in the background (parallel across datasets)

This is a 1–5 hour process. Launch each dataset with the Bash tool using **`run_in_background: true`** and
capture a log. Then **stop** — do not sleep/poll on a timer. The harness re-invokes you when a process
exits; proceed to the cross-dataset success check (§4.6) and Stage 5 only after **all** launched processes
have exited.

**One independent background process per dataset**, and run them **in parallel on the single rag-server**
when the server has enough workers. Each process writes its own snapshot folder and log, so failures and
timestamps stay isolated.

### Step 1 — Confirm the rag-server worker count (gate on ≥16)

Parallel eval is safe only when rag-server runs with **≥16 workers**. The Compose default is `--workers 16`
(`deploy/compose/docker-compose-rag-server.yaml`). Read the live value — do not assume:

```bash
# Live container (authoritative): inspect the actual start command
docker inspect rag-server --format '{{join .Config.Cmd " "}}' 2>/dev/null   # look for: --workers <N>
# Fallback: the compose definition
grep -nE '\-\-workers' deploy/compose/docker-compose-rag-server.yaml
```

- **N ≥ 16** → parallel fan-out is allowed.
- **N < 16, or undetermined** → run **sequentially** (one dataset at a time; wait for each to finish and
  complete Stage 5 before the next). Optionally tell the user they can raise `--workers` to 16+ and redeploy
  rag-server (via the `rag-blueprint` skill) to unlock parallel runs.

### Step 2 — Decide the parallelism limit (default 4)

- **Default: at most 4 datasets in parallel.** This balances throughput against per-request latency on a
  16-worker server.
- **User override:** if the user specifies a parallelism in their prompt (e.g. "run 6 in parallel",
  "max 2 at a time"), **use that value** and **emit a warning** that states the chosen value and the trade-off.
  Examples:
  - Above 4: *"⚠️ Running 6 datasets in parallel as requested — above the recommended max of 4. On a
    16-worker server this raises per-request latency and the chance of timeouts; latency metrics become less
    comparable to sequential baselines."*
  - Below the dataset count but ≥1: *"Running at most 2 datasets in parallel as requested; remaining datasets
    queue and start as slots free up."*
- The effective concurrency is `min(requested_or_default_limit, num_datasets)`. If more datasets than the
  limit are selected, launch the first `limit` now and start each remaining dataset as a running one exits.

### Step 3 — Split `--thread` so combined load ≈ worker count

Each process issues up to `--thread` concurrent requests. Running `P` datasets in parallel multiplies that
to `P × --thread`. Keep the **total** near the worker count so you don't oversubscribe:

```
per_process_thread ≈ max(4, floor(workers / P))     # e.g. 16 workers, P=4 → --thread 4
```

Use this per-process `--thread` for every parallel run. (Sequential runs keep the usual `--thread 16`.)

### Step 4 — Launch one process per dataset

Compute each cycle's snapshot folder **first** and `tee` the log straight into it, so the log lands next to
the results it describes. `CYCLE` is `baseline` for the first run of a dataset, then `cycle1`/`cycle2`/`cycle3`
for the Stage 6.7 improve cycles. Keep `--output_dir results` — each process passes a single `--datasets
"$DS"`, so the script writes to its own `results/$DS/` subdir and **parallel runs never collide** (different
datasets → different subdirs):

```bash
cd "$SCRIPTS_DIR" && source .venv/bin/activate && export NVIDIA_API_KEY="$NVIDIA_API_KEY"
DS=financebench; CYCLE=baseline
TS=$(date +%Y%m%d_%H%M%S)
SNAP="$EXP_DIR/$DS/${CYCLE}_${TS}"; mkdir -p "$SNAP"
PYTHONUNBUFFERED=1 python3 evaluate_rag.py <flags from 4.3> \
  --datasets "$DS" --thread <per_process_thread> --output_dir results 2>&1 | tee "$SNAP/eval.log"
```

Issue **one such `run_in_background: true` Bash call per dataset** (up to the parallelism limit). Note each
`$SNAP/eval.log` path so Stage 5 can read the log and copy the JSONs from `results/$DS/`.

> **Ingestion contends on a single-worker ingestor.** rag-server has 16 workers, but the **ingestor** runs
> with `--workers 1` by default, so concurrent ingestion across datasets serializes and can time out. The
> parallelism win is on the **generation/eval** phase, not ingestion. When fanning out, prefer to **pre-ingest**
> (run each dataset once with `--skip_evaluation` to populate its collection, sequentially or lightly
> staggered), then launch the parallel eval with `--skip_ingestion`. Each dataset uses a **distinct
> collection**, so parallel runs never clobber each other's data.

Guidance:
- Never run two evals of the **same** dataset at once — they share one collection and will corrupt it.
- If you must set a fallback wakeup, make it long (≥1200 s) — the exit notification is the real signal.
- Proceed to §4.6, then Stage 5, only after **every** launched process has exited.

## 4.5 What "done" looks like

On a successful exit you will see the `RAG Evaluation` banner, ingestion metrics, `EVALUATION RESULTS`
(End-2-End Accuracy, context_relevance, response_groundedness, recall, token usage, and a **Latency (per
query)** block — samples, mean/p50/p90/p99/min/max total seconds, mean/p90 TTFT), a
`Stream token breakdown (N samples) written to …` line (the per-stage token-cost report), and
`Evaluation complete. Results stored in directory: results`. Proceed to §4.6 (and then Stage 5) to verify
the files on disk regardless of what the console said.

## 4.6 After all runs exit — verify every query succeeded (across all datasets)

When running datasets in parallel, the harness re-invokes you **once per process exit**. Do the per-run
output validation in Stage 5 as usual, but **only after the last dataset's process has exited**, run this
**cross-dataset query-success check** so a partial failure in any one dataset is caught before analysis.

For each dataset, a query is "successful" when it produced a non-empty `generated_answer` and did not error.
Compare the answered count to the dataset's expected query count and flag any dataset with failures:

```bash
cd "$SCRIPTS_DIR"
echo "=== Cross-dataset query-success check ==="
overall_ok=1
for SNAP in "$EXP_DIR"/*/baseline_* "$EXP_DIR"/*/cycle*_*; do
  [ -d "$SNAP" ] || continue
  DS=$(basename "$(dirname "$SNAP")")
  DATA=$(ls "$SNAP"/rag_*_evaluation_data.json 2>/dev/null | head -1)
  if [ -z "$DATA" ]; then
    echo "✗ $DS ($SNAP): no evaluation_data.json — run did not complete"; overall_ok=0; continue
  fi
  total=$(jq 'length' "$DATA")
  # failed = empty/missing generated_answer
  failed=$(jq '[.[] | select((.generated_answer // "") | gsub("\\s";"") == "")] | length' "$DATA")
  ok=$((total - failed))
  if [ -f "$SNAP/failure.txt" ] || [ "$failed" -gt 0 ]; then
    echo "✗ $DS: $ok/$total queries OK, $failed failed${SNAP:+  ($SNAP)}"; overall_ok=0
  else
    echo "✓ $DS: $ok/$total queries OK"
  fi
done
[ "$overall_ok" -eq 1 ] && echo "ALL DATASETS: every query succeeded." \
                        || echo "ATTENTION: one or more datasets had failed queries — diagnose (Stage 5.2) before analysis."
```

- **All green** → proceed to Stage 5 (snapshot) and Stage 6 (analysis) for the full set.
- **Any dataset with failures** (`failure.txt`, missing data file, or `failed > 0`) → treat **that dataset's
  run** as failed: diagnose from its `eval.log` and retry it **once** per Stage 5.2 (the healthy datasets'
  results stand and proceed normally). Report the per-dataset OK/failed counts to the user.

Record the OK/failed counts per dataset in `state.md` alongside the metrics so the experiment manifest shows
coverage at a glance.
