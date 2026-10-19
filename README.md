# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Author:** Sparsh Bhartia  
**Course:** Language Models and Applications (LMA)  
**Deliverables:** [Jupyter Notebook (`RAG_Assignment.ipynb`)](RAG_Assignment.ipynb) | [PDF Report (`REPORT.pdf`)](REPORT.pdf) | [Markdown Report (`REPORT.md`)](REPORT.md)

---

## Overview

In this project, I implemented and evaluated a **Two-Stage Retrieval-Augmented Generation (RAG)** pipeline to compare baseline dense vector retrieval against cross-encoder re-ranking on a complex, 403-page rulebook (*Dungeons & Dragons 5th Edition System Reference Document - SRD 5.1*).

### The Two Pipelines
1. **Pipeline 1 (Baseline Retrieval):**
   - Chunks the document into ~400 token passages (50 token overlap).
   - Indexes all 1,094 chunks into a FAISS `IndexFlatIP` vector store using `all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Retrieves the Top 10 chunks, feeds the Top 3 directly to Gemini (`gemini-2.0-flash`), and generates **Answer 1**.
2. **Pipeline 2 (Two-Stage Re-Ranking):**
   - Takes the Top 10 candidate chunks retrieved by the Bi-Encoder.
   - Pairs the query with each candidate and scores all 10 pairs using a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - Re-sorts by score, selects the Top 3 re-ranked chunks, and feeds them to Gemini to generate **Answer 2**.

---

## Key Results: Failure Mode Stress-Testing

I designed four realistic stress-test queries where the baseline Bi-Encoder failed to rank the ground-truth chunk in the Top 3, but the Cross-Encoder successfully promoted it:

| Category | Query | Gold Chunk | Page | Baseline Rank | Re-Ranked Rank | Rank Shift | Baseline Verdict | Re-Ranked Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Keyword Trap** | *Can a rogue use sneak attack with a heavy two-handed greatsword?* | #157 / #87 | 66 / 39 | #4 / #6 | #1 / #2 | **#4 $\rightarrow$ #1**<br>**#6 $\rightarrow$ #2** | **Hallucination:** Matched weapon stats; missed finesse rule. Stated sneak attack works with advantage. | **Correct:** Promoted sneak attack rule; clearly stated greatsword cannot sneak attack. |
| **Exception / Corner Case** | *What is the minimum Con saving throw DC to maintain concentration when taking damage?* | #988 | 357 | #9 | #2 | **#9 $\rightarrow$ #2** | **Omission:** Retrieved general saving throw chunks; missed the DC 10 floor rule. | **Correct:** Promoted concentration threshold chunk; gave exact formula (DC 10 or half damage). |
| **Negation & Constraints** | *What actions or reactions can a creature take while under the Incapacitated condition?* | #993 | 359 | #4 | #1 | **#4 $\rightarrow$ #1** | **Omission:** Matched generic action/reaction rules; missed the condition prohibition. | **Correct:** Promoted Incapacitated definition; correctly stated no actions or reactions are allowed. |
| **Buried Chunk** | *What specific rules and attack penalties apply to melee attacks while fighting underwater?* | #239 | 95 | #7 | #2 | **#7 $\rightarrow$ #2** | **Omission:** Dominated by generic 5-foot reach chunks; missed disadvantage penalty. | **Correct:** Promoted underwater combat rules; identified disadvantage and weapon exceptions. |

---

## Latency & Scaling Summary

- **Latency:** On CPU, Bi-Encoder retrieval took **~11 ms**, while Cross-Encoder scoring of 10 pairs took **~650 ms** (~58x overhead). However, in production with an online LLM (which takes 800–2,000 ms), adding 5–10 ms of GPU re-ranking is negligible and eliminates hallucinations.
- **Why Not Cross-Encode Everything?** Scoring 1,094 chunks with a pure cross-encoder would take **~71 seconds per query** on CPU, and on a 1,000,000 chunk corpus it would take **~18 hours per query**. The two-stage funnel (Bi-Encoder filters to 10 $\rightarrow$ Cross-Encoder picks top 3) provides the optimal trade-off between speed and accuracy.

---

## Repository Structure

```
├── RAG_Assignment.ipynb       # Main end-to-end Jupyter Notebook with all stages
├── REPORT.pdf                 # Formatted 3-page technical report (PDF)
├── REPORT.md                  # Markdown version of technical report
├── report.tex                 # LaTeX source code for report
├── dnd_srd.pdf                # Source document (D&D 5e SRD 5.1)
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and instructions
```

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Sparsh181/RAG-Baseline-vs-ReRanking.git
cd RAG-Baseline-vs-ReRanking
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

### 3. Run the Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook RAG_Assignment.ipynb
```
Run all cells from top to bottom.
