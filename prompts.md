# Prompts Used (Academic Integrity Log)

This file logs all AI assistant prompts used in building Lab 24. Required per section 7.2 and FAQ Q8.

---

## Phase A — RAGAS Evaluation

**Prompt 1 — Generate test set structure**
```
Generate a 54-question test set from two Vietnamese documents: Nghị định 13/2023/NĐ-CP (personal data protection) and a BCTC tax return (bctc.md). Distribute questions as ~50% simple fact retrieval, ~27% reasoning/multi-step, ~22% multi-context. Output JSON with fields: question_id, question, ground_truth, question_type, source_doc.
```

**Prompt 2 — RAGAS metric interpretation**
```
My RAGAS scores are: faithfulness=0.724, answer_relevancy=0.722, context_precision=0.958, context_recall=0.805. Targets are F≥0.85, AR≥0.80, CP≥0.70, CR≥0.75. Identify which metrics are below target and suggest root causes for a Vietnamese RAG pipeline using top-3 chunk retrieval with Qdrant hybrid search.
```

**Prompt 3 — Failure cluster analysis**
```
Given a CSV of RAGAS results with columns (question_id, faithfulness, answer_relevancy, context_precision, context_recall), write a Python script that identifies failure rows (any metric below threshold: F<0.5, AR<0.5, CP<0.5, CR<0.5) and clusters them into at most 4 clusters by dominant failure type. Output a Markdown failure_analysis.md with cluster definitions and actionable recommendations.
```

---

## Phase B — LLM-as-Judge

**Prompt 4 — Pairwise judge prompt design**
```
Write a pairwise LLM-as-Judge prompt for Vietnamese RAG evaluation. Given a question, Answer A (RAG-generated), and Answer B (raw retrieved context), the judge should pick which answer better serves the user. Include swap-and-average bias mitigation: run the evaluation twice with A and B swapped, then reconcile. Output winner as JSON: {"winner": "A"|"B"|"tie"}.
```

**Prompt 5 — Absolute scoring rubric**
```
Design a 4-dimension absolute scoring rubric (1-5 scale) for Vietnamese RAG answers: accuracy (factual correctness), relevance (addresses the question), conciseness (no unnecessary verbosity), helpfulness (user can act on the answer). Return JSON with dimension scores and a brief justification for each.
```

**Prompt 6 — Cohen's kappa analysis**
```
I have 10 human labels and 10 LLM judge labels for the same pairwise comparisons (labels are A, B, or tie). Calculate Cohen's kappa and explain the agreement level. Identify systematic bias patterns (position bias, length bias, tie-over-conservatism).
```

---

## Phase C — Guardrails Stack

**Prompt 7 — PII detection chain design**
```
Design a PII detection chain for Vietnamese text that handles: CCCD (12-digit national ID), Vietnamese phone numbers (10-digit starting 03/05/07/08/09), email addresses, tax codes (10-13 digits), and PERSON names. Combine rule-based VN regex with Microsoft Presidio NER. Return detected entities with type, value, start/end positions, and a redacted version of the input text.
```

**Prompt 8 — Topic scope validator**
```
Build a topic scope validator for a Vietnamese RAG assistant scoped to: (1) Nghị định 13/2023/NĐ-CP personal data protection law, and (2) BCTC corporate tax returns. Use sentence-transformers to compute cosine similarity between the user query and topic anchors. If max similarity < 0.45, return a polite Vietnamese refusal message. Keep latency under 50ms for steady-state (warm model).
```

**Prompt 9 — Adversarial attack test suite**
```
Generate 20 adversarial attack prompts targeting a Vietnamese RAG guardrail system. Include: DAN jailbreak variants, roleplay persona injection, payload splitting across multiple messages, Base64/Unicode encoding obfuscation, indirect prompt injection via retrieved documents, and social engineering. For each attack, specify the attack type and expected blocked/allowed outcome.
```

**Prompt 10 — Llama Guard integration**
```
Integrate Llama Guard 3 as an output safety classifier for a Vietnamese RAG pipeline. Support three provider modes: (1) HuggingFace API with HF_TOKEN, (2) LM Studio local via OpenAI-compatible API, (3) rule-based fallback using keyword lists. Return {"safe": true/false, "category": str, "provider": str}. Log all flagged outputs to an async JSONL audit log.
```

**Prompt 11 — Async guardrail pipeline**
```
Refactor the guardrail stack to run L1 (PII + topic validation) checks in parallel using asyncio.gather, and run L3 (Llama Guard output check) asynchronously after generation. Benchmark latency with 100 requests and report P50/P95/P99 for each layer. Target: L1 < 50ms P50, total pipeline < 600ms P95.
```

---

## Phase D — Production Blueprint

**Prompt 12 — SLO table design**
```
Define 7 production SLOs for a Vietnamese RAG evaluation and guardrail system. For each SLO include: metric name, target value, measurement method, and alert threshold. Cover: faithfulness, answer relevancy, PII detection rate, adversarial block rate, L1 latency P95, total pipeline latency P95, guardrail uptime.
```

**Prompt 13 — Architecture diagram**
```
Write a Mermaid diagram showing the full production architecture: user request → L1 guardrails (PII redaction, topic validation) → RAG pipeline (Qdrant hybrid search + reranking + LLM generation) → L3 output guard (Llama Guard) → L4 audit log → response to user. Include async paths and fallback providers.
```

**Prompt 14 — Alert playbooks**
```
Write 3 operational alert playbooks for the Vietnamese RAG system: (1) faithfulness score drops below 0.70 in a 1-hour window, (2) L1 guardrail latency P95 exceeds 200ms, (3) PII detection rate drops below 90%. Each playbook should include: trigger condition, immediate diagnostic steps, likely root causes, and remediation actions.
```

**Prompt 15 — Cost analysis**
```
Estimate monthly infrastructure cost for a Vietnamese RAG guardrail system serving 100,000 queries/month. Break down costs by: LLM API calls (OpenAI GPT-4o-mini for generation + judging), embedding API (text-embedding-3-small), Qdrant Cloud storage, compute for async guardrail workers, and monitoring. Compare API-based vs self-hosted options.
```

---

## Scaffold Adaptation

**Prompt 16 — Lab 18 to Lab 24 migration**
```
The existing Lab 18 codebase has a Vietnamese RAG pipeline with Qdrant hybrid search, reranking, and OpenAI/LM Studio generation. Adapt the scaffold for Lab 24 which adds: RAGAS evaluation (phase-a/), LLM-as-Judge pairwise comparison (phase-b/), defense-in-depth guardrails (phase-c/), and a production blueprint document (phase-d/). Keep the Day 18 RAG backend intact as the retrieval/generation engine. Add config.py for centralized settings, main.py as the unified CLI entry point, and check_lab.py for submission validation.
```
