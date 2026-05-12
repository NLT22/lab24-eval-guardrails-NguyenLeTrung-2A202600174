# Lab 24 - Full Evaluation & Guardrail System

This repo implements the Lab 24 evaluation and guardrail stack for a Vietnamese
RAG pipeline. The Day 18 RAG code is kept as the retrieval/generation backend,
while Lab 24 artifacts live under `phase-a/`, `phase-b/`, `phase-c/`, and
`phase-d/`.

## Setup

Use Python 3.12 if possible.

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
cp .env.example .env
```

For local vector search with Qdrant:

```bash
docker compose up -d
```

## Environment

The project can run in cloud mode or local-first mode.

Cloud keys:

```env
OPENAI_API_KEY=
HF_TOKEN=
HUGGINGFACEHUB_API_TOKEN=
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=lab24-eval-guardrails
```

LM Studio fallback:

```env
LLM_PROVIDER=auto
EMBEDDING_PROVIDER=auto
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=<chat-model-loaded-in-lm-studio>
LMSTUDIO_EMBEDDING_MODEL=<embedding-model-loaded-in-lm-studio>
LMSTUDIO_GUARD_MODEL=<llama-guard-model-loaded-in-lm-studio>
```

Output guard:

```env
OUTPUT_GUARD_PROVIDER=auto
LLAMA_GUARD_MODEL=meta-llama/Llama-Guard-3-8B
```

`auto` uses OpenAI/HuggingFace when keys are available, otherwise falls back to
LM Studio or local deterministic guards where implemented.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── prompts.md
├── .env.example
├── config.py
├── check_lab.py
├── docker-compose.yml
├── test_set.json
├── data/
├── src/                 # RAG backend: chunking, search, rerank, eval, pipeline
├── tests/
├── phase-a/             # RAGAS evaluation artifacts and scripts
├── phase-b/             # LLM-as-judge artifacts
├── phase-c/             # Guardrails stack
├── phase-d/             # Production blueprint
└── demo/
```

## Phase A - RAGAS Evaluation

Scripts:

- `phase-a/generate_testset.py`
- `phase-a/run_ragas_eval.py`
- `phase-a/analyze_failures.py`

Fast/offline run using existing evaluation report:

```bash
python phase-a/generate_testset.py
python phase-a/run_ragas_eval.py --mode existing
python phase-a/analyze_failures.py
```

Live run through the RAG pipeline and RAGAS:

```bash
python phase-a/generate_testset.py
python phase-a/run_ragas_eval.py --mode live --limit 5
python phase-a/analyze_failures.py
```

Remove `--limit 5` to evaluate the full test set.

To log eval runs to LangSmith, set this in `.env` before running Phase A/B:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your-langsmith-key>
LANGSMITH_PROJECT=lab24-eval-guardrails
```

The scripts also set older LangChain-compatible variables automatically
(`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`) for RAGAS and
LangChain integrations.

Outputs:

- `phase-a/testset_v1.csv`
- `phase-a/testset_review_notes.md`
- `phase-a/ragas_results.csv`
- `phase-a/ragas_summary.json`
- `phase-a/failure_analysis.md`

## Phase B - LLM-as-Judge

Expected artifacts:

- `phase-b/pairwise_results.csv`
- `phase-b/absolute_scores.csv`
- `phase-b/human_labels.csv`
- `phase-b/kappa_analysis.py` or notebook
- `phase-b/judge_bias_report.md`

Judge prompts should use `src/llm_client.py`, which supports OpenAI and LM
Studio through the same OpenAI-compatible interface.

Run Phase B after Phase A has produced `phase-a/ragas_results.csv`:

```bash
python phase-b/judge.py --limit 30
```

This creates:

- `phase-b/pairwise_results.csv`
- `phase-b/absolute_scores.csv`
- `phase-b/judge_bias_report.md`

Then create/update the human calibration file and compute Cohen's kappa:

```bash
python phase-b/kappa_analysis.py
```

The first run creates `phase-b/human_labels.csv` if it does not exist. Edit
that file manually, fill `human_winner` with `A`, `B`, or `tie`, then rerun:

```bash
python phase-b/kappa_analysis.py
```

Output:

- `phase-b/human_labels.csv`
- `phase-b/kappa_report.md`

If LM Studio is slow, use fewer rows first:

```bash
python phase-b/judge.py --limit 5
python phase-b/kappa_analysis.py
```

## Phase C - Guardrails Stack

Current code:

- `phase-c/topic_guard.py`: embedding/keyword topic scope validator.

Run topic guard smoke test:

```bash
python phase-c/topic_guard.py
```

Expected Phase C artifacts:

- `phase-c/input_guard.py`
- `phase-c/output_guard.py`
- `phase-c/full_pipeline.py`
- `phase-c/pii_test_results.csv`
- `phase-c/adversarial_test_results.csv`
- `phase-c/latency_benchmark.csv`

For output safety, use HuggingFace Llama Guard 3 when `HF_TOKEN` is available,
or load a Llama Guard model in LM Studio and set `LMSTUDIO_GUARD_MODEL`.

## Phase D - Blueprint

Expected artifact:

- `phase-d/blueprint.md`

The blueprint should include SLOs, architecture diagram, alert playbooks, and
monthly cost analysis.

## Validation

Run tests:

```bash
python -m pytest tests -q
```

Run submission checker:

```bash
python check_lab.py
```

## Results Summary

Current Phase A summary:

```text
Faithfulness:       0.7242
Answer Relevancy:  0.7224
Context Precision: 0.9583
Context Recall:    0.8052
```

See `phase-a/ragas_summary.json` and `phase-a/failure_analysis.md` for details.

## Notes

- `OPEN_API_KEY` is accepted as a compatibility alias, but use
  `OPENAI_API_KEY` in `.env`.
- `HF_TOKEN` and `HUGGINGFACEHUB_API_TOKEN` are both supported; `HF_TOKEN` is
  preferred.
- If pip cannot find packages, check for environment variables like
  `PIP_NO_INDEX`, `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`.
