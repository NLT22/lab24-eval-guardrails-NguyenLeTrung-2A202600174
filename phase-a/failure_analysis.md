# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question | F | AR | CP | CR | Avg | Cluster |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Kỳ tính thuế của tờ khai GTGT trong file bctc.md là gì? | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | C1 |
| 2 | Tờ khai thuế GTGT trong file bctc.md được nộp lần đầu hay không? | 0.000 | 0.000 | 1.000 | 1.000 | 0.500 | C1 |
| 3 | Thuế GTGT phát sinh trong kỳ (129.511.633 đồng) và thuế GTGT được khấu trừ (77.377.803 đồn | 0.000 | 0.748 | 1.000 | 0.500 | 0.562 | C1 |
| 4 | Thuế giá trị gia tăng được khấu trừ (chỉ tiêu [22]) trong kỳ Quý 4 năm 2024 là bao nhiêu? | 0.000 | 0.849 | 0.500 | 1.000 | 0.587 | C1 |
| 5 | Theo Nghị định 13/2023/NĐ-CP, sự im lặng của chủ thể dữ liệu có được coi là sự đồng ý xử l | 1.000 | 0.000 | 1.000 | 0.500 | 0.625 | C2 |
| 6 | Tờ khai thuế GTGT trong bctc.md áp dụng theo mẫu số nào? | 0.000 | 0.765 | 1.000 | 1.000 | 0.691 | C1 |
| 7 | Thuế GTGT đề nghị hoàn (chỉ tiêu [42]) trong kỳ Quý 4 năm 2024 là bao nhiêu? | 0.000 | 0.818 | 1.000 | 1.000 | 0.704 | C1 |
| 8 | Người ký tờ khai thuế GTGT Quý 4 năm 2024 là ai? | 1.000 | 0.846 | 0.500 | 0.500 | 0.712 | C3 |
| 9 | Dữ liệu cá nhân nhạy cảm theo Nghị định 13/2023/NĐ-CP gồm những loại thông tin nào? | 0.667 | 0.871 | 0.500 | 1.000 | 0.759 | C3 |
| 10 | Điều 8 Nghị định 13/2023/NĐ-CP nghiêm cấm những hành vi nào? | 1.000 | 0.840 | 1.000 | 0.200 | 0.760 | C4 |

## Clusters Identified

### C1: Faithfulness / hallucination failures

**Pattern:** Related metric is the weakest signal in the bottom questions.

**Examples:**
- Kỳ tính thuế của tờ khai GTGT trong file bctc.md là gì?
- Tờ khai thuế GTGT trong file bctc.md được nộp lần đầu hay không?

**Proposed fix:** Tune retrieval/reranking and tighten answer grounding prompts. For low recall, increase top-k or use parent chunks; for low precision, add metadata filters.

### C2: Answer relevancy failures

**Pattern:** Related metric is the weakest signal in the bottom questions.

**Examples:**
- Theo Nghị định 13/2023/NĐ-CP, sự im lặng của chủ thể dữ liệu có được coi là sự đồng ý xử lý dữ liệu cá nhân không?

**Proposed fix:** Tune retrieval/reranking and tighten answer grounding prompts. For low recall, increase top-k or use parent chunks; for low precision, add metadata filters.

### C3: Irrelevant retrieval context

**Pattern:** Related metric is the weakest signal in the bottom questions.

**Examples:**
- Người ký tờ khai thuế GTGT Quý 4 năm 2024 là ai?
- Dữ liệu cá nhân nhạy cảm theo Nghị định 13/2023/NĐ-CP gồm những loại thông tin nào?

**Proposed fix:** Tune retrieval/reranking and tighten answer grounding prompts. For low recall, increase top-k or use parent chunks; for low precision, add metadata filters.

### C4: Missing context / low recall

**Pattern:** Related metric is the weakest signal in the bottom questions.

**Examples:**
- Điều 8 Nghị định 13/2023/NĐ-CP nghiêm cấm những hành vi nào?

**Proposed fix:** Tune retrieval/reranking and tighten answer grounding prompts. For low recall, increase top-k or use parent chunks; for low precision, add metadata filters.
