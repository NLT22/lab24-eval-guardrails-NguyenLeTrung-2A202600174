# Judge Bias Report

## Position Bias

- A wins when listed first: 27/30 (90.0%).
- Mitigation used: swap-and-average; final winner becomes tie if the two orders disagree.

## Length Bias

- Longer answer wins: 1/27 (3.7%).

## Final Winner Distribution

| Winner | Count |
|---|---:|
| A | 26 |
| B | 1 |
| tie | 3 |

## Mitigation Strategy

- Keep swap-and-average for pairwise comparisons.
- Add rubric-based absolute scoring for monitoring.
- Manually calibrate with Cohen's kappa before trusting judge scores in CI.
