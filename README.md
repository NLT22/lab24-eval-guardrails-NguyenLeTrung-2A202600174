# Lab 24 — Full Evaluation & Guardrail System

## Overview

This repo implements a production-ready evaluation and guardrail stack for a Vietnamese RAG pipeline built in Day 18. The system covers four phases: automated RAGAS evaluation with failure cluster analysis (Phase A), LLM-as-Judge pairwise comparison with human calibration (Phase B), a defense-in-depth guardrail stack with PII redaction, topic validation, adversarial testing, and Llama Guard output safety (Phase C), and a production blueprint document with SLOs, architecture diagram, alert playbooks, and cost analysis (Phase D). The Day 18 RAG backend (Qdrant hybrid search + reranking) is kept as the retrieval/generation engine while Lab 24 artifacts live under `phase-a/` through `phase-d/`.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (or OPEN_API_KEY alias)
```

For local vector search with Qdrant:

```bash
docker compose up -d
```

Environment variables needed:

```env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...             # for Llama Guard via HuggingFace
OUTPUT_GUARD_PROVIDER=auto  # auto | hf | lmstudio | rule_based
```

LM Studio fallback (no API key needed):

```env
LLM_PROVIDER=auto
EMBEDDING_PROVIDER=auto
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=<chat-model>
LMSTUDIO_GUARD_MODEL=<llama-guard-model>
```

## Results Summary

### Phase A — RAGAS Evaluation

- **Test set:** 54 questions from Vietnamese corpus (Nghị định 13/2023/NĐ-CP + BCTC tax return) with distribution: ~50% simple, ~27% reasoning, ~22% multi-context
- **RAGAS scores (54-question aggregate from pre-computed report):**

  | Metric | Score | Target | Status |
  |---|---:|---:|---|
  | Faithfulness | 0.724 | ≥ 0.85 | below target |
  | Answer Relevancy | 0.722 | ≥ 0.80 | below target |
  | Context Precision | 0.958 | ≥ 0.70 | ✓ |
  | Context Recall | 0.805 | ≥ 0.75 | ✓ |

- **Live run scores (33 questions):** F=0.72 | AR=0.76 | CP=0.89 | CR=0.87
- **Total eval cost:** ~$0 (LM Studio local fallback used throughout)
- **Failure clusters identified:** 4 clusters (see [phase-a/failure_analysis.md](phase-a/failure_analysis.md))
  - C1: Faithfulness/hallucination failures (6 questions — model adds unretrieved details)
  - C2: Answer relevancy failures (1 question — answer technically correct but doesn't directly address question)
  - C3: Irrelevant retrieval context / low context precision (2 questions — retriever returns mixed-document chunks)
  - C4: Missing context / low recall (1 question — corpus doesn't cover all sub-facts in ground truth)

**Faithfulness and Answer Relevancy below 0.85/0.80 targets.** Root cause: RAG pipeline uses top-3 chunks; for multi-step questions this is insufficient. Fix: increase top-k to 5, add parent-chunk retrieval, tighten evidence-only generation prompt.

### Phase B — LLM-as-Judge

- **Pairwise judging:** 30 questions judged with swap-and-average bias mitigation (each pair run twice with swapped order; tie if judges disagree)
- **Cohen's kappa vs human:** 0.583 (moderate agreement — borderline production-ready)
- **Root cause of kappa < 0.6:** Judge applies conservative tie policy when swapped runs disagree; human labels 8/10 as A (RAG answer clearly better than raw context B). Judge labeled 3 ties where human labeled A — swap-conservatism bias.
- **Position bias:** measured via run1 vs run2 winner distribution (see [phase-b/judge_bias_report.md](phase-b/judge_bias_report.md))
- **Length bias:** B (raw context) wins more when it is longer; mitigated by swap-and-average
- **Absolute scores:** 30 questions scored on 4-dimension rubric (accuracy, relevance, conciseness, helpfulness) — saved in [phase-b/absolute_scores.csv](phase-b/absolute_scores.csv)

### Phase C — Guardrails Stack

- **PII detection:** 7/7 PII inputs detected (100%) — mix of CCCD, phone_vn, email, PERSON entities; rule-based VN regex + Presidio NER chain
- **Topic validator:** embedding cosine similarity (sentence-transformers fallback); threshold 0.45; graceful Vietnamese refusal message
- **Adversarial defense:** 18/20 attacks blocked (90%) — DAN, roleplay, payload splitting, encoding, indirect injection variants
- **Output guard:** rule-based fallback (Llama Guard 3 via HuggingFace/LM Studio when available); async L4 audit log to `phase-c/audit_log.jsonl`
- **Latency benchmark (100 requests):**

  | Layer | P50 | P95 |
  |---|---:|---:|
  | L1 Input guards | 15 ms | 424 ms |
  | L3 Output guard | 512 ms | 534 ms |
  | Total end-to-end | 521 ms | 551 ms |

  L1 P95 spike driven by cold-start embedding model load; steady-state P50 = 15 ms (within <50ms target). L3 dominated by local LM Studio inference.

### Phase D — Blueprint

See [phase-d/blueprint.md](phase-d/blueprint.md) — includes SLO table (7 metrics), Mermaid architecture diagram, 3 alert playbooks (faithfulness drop, latency spike, guardrail detection drop), and monthly cost analysis (~$386/month at 100k queries).

## Lessons Learned

**RAGAS reveals what demos hide.** Running live RAGAS on 33 questions surfaced concrete failure patterns that would have been invisible from manual inspection: faithfulness=0 on questions where the model added plausible but unretrieved details, and answer_relevancy=0 when the answer was technically accurate but phrased around a tangent. The failure cluster analysis (4 distinct clusters) gives a prioritized roadmap: fix faithfulness first (largest cluster, clearest fix via evidence-only prompt), then precision (top-k tuning).

**LLM judges are useful but conservatively biased when using swap-and-average.** The swap-and-average strategy correctly mitigates position bias, but it produces more ties than a human would: when the two ordered runs give different winners, defaulting to "tie" is conservative. Human kappa of 0.583 reflects this — humans preferred Answer A (RAG answer) in 8/10 cases while the judge only confirmed 6. For production, consider a weighted consensus rule (e.g., count "almost-tie" as the majority) to reduce over-conservative ties.

**Async guardrails are necessary, not optional.** Running L1 (PII + topic) and L3 (Llama Guard) synchronously in sequence would add 500–1000ms per request. The async parallel architecture keeps total overhead manageable. The real bottleneck is cold-start time for local embedding/guard models — in production, keep models warm with a keepalive ping or use API-based guards (Groq Llama Guard free tier) to avoid cold starts entirely.

## Demo Video

*(Record a 5-minute demo showing: RAGAS live on 5 questions, LLM-Judge comparison, 3 adversarial attacks blocked, latency benchmark P50/P95/P99)*

https://youtu.be/rfvz6LBXR7w

## Notes

- `OPEN_API_KEY` is accepted as a compatibility alias for `OPENAI_API_KEY`.
- `HF_TOKEN` and `HUGGINGFACEHUB_API_TOKEN` are both supported.
- Output guard falls back to rule-based if neither HuggingFace nor LM Studio guard model is available.
- Run `python check_lab.py` for submission validation.
