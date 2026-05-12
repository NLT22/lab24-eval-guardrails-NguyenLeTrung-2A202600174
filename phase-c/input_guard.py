"""Phase C.1 - Input guardrail: PII redaction + tests."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass


VN_PII = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"\b(?:\+84|0)\d{9,10}\b",
    "tax_code": r"\b\d{10}(?:-\d{3})?\b",
    "email": r"\b[\w.\-+]+@[\w.\-]+\.\w+\b",
}


@dataclass
class GuardResult:
    sanitized: str
    latency_ms: float
    detected_types: list[str]


class InputGuard:
    def __init__(self) -> None:
        self.analyzer = None
        self.anonymizer = None
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except Exception:
            # Regex fallback is enough for local smoke tests.
            self.analyzer = None
            self.anonymizer = None

    def scrub_vn(self, text: str) -> tuple[str, list[str]]:
        detected: list[str] = []
        out = text or ""
        for name, pattern in VN_PII.items():
            if re.search(pattern, out):
                detected.append(name)
            out = re.sub(pattern, f"[{name.upper()}]", out)
        return out, detected

    def scrub_ner(self, text: str) -> tuple[str, list[str]]:
        if not self.analyzer or not self.anonymizer or not text:
            return text, []
        try:
            results = self.analyzer.analyze(text=text, language="en")
            anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results).text
            return anonymized, sorted({r.entity_type for r in results})
        except Exception:
            return text, []

    def sanitize(self, text: str) -> GuardResult:
        start = time.perf_counter()
        out, detected_regex = self.scrub_vn(text or "")
        out, detected_ner = self.scrub_ner(out)
        latency_ms = (time.perf_counter() - start) * 1000
        return GuardResult(out, latency_ms, detected_regex + detected_ner)

    async def sanitize_async(self, text: str) -> GuardResult:
        return self.sanitize(text)


def test_inputs() -> list[tuple[str, bool]]:
    return [
        ("Hi, I'm John Smith from Microsoft. Email: john@ms.com", True),
        ("Call me at +1-555-1234 or visit 123 Main Street, NYC", True),
        ("So CCCD cua toi la 012345678901", True),
        ("Lien he qua 0987654321 hoac tax 0123456789-001", True),
        ("Customer Nguyen Van A, CCCD 098765432101, phone 0912345678", True),
        ("", False),
        ("Just a normal question about Decree 13", False),
        ("A" * 5000, False),
        ("Le Van Binh o 123 Le Loi", True),
        ("tax_code:0123456789-001 cccd:012345678901", True),
    ]


def evaluate(path: str = "phase-c/pii_test_results.csv") -> dict[str, float]:
    guard = InputGuard()
    rows = []
    true_positive = 0
    positives = 0
    latencies = []
    for text, expected_pii in test_inputs():
        result = guard.sanitize(text)
        detected = bool(result.detected_types) or result.sanitized != text
        if expected_pii:
            positives += 1
            true_positive += int(detected)
        latencies.append(result.latency_ms)
        rows.append({
            "input": text[:120],
            "expected_pii": expected_pii,
            "detected": detected,
            "detected_types": ";".join(result.detected_types),
            "sanitized": result.sanitized[:160],
            "latency_ms": f"{result.latency_ms:.3f}",
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"recall": true_positive / positives if positives else 0.0, "p95_ms": percentile(latencies, 95)}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="phase-c/pii_test_results.csv")
    args = parser.parse_args()
    result = evaluate(args.out)
    print(f"PII recall={result['recall']:.1%}, P95={result['p95_ms']:.2f}ms")


if __name__ == "__main__":
    main()
