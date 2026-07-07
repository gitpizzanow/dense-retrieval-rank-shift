# Experimental Workflow: Dense Retrieval Dimensionality

This document outlines the code sections inside `conf (5).ipynb` and explains the scientific narrative behind the experiments. It details why we started with one approach, why we pivoted, and how we arrived at our final conclusions for the IEEE paper.

---

### ![STEP 1](https://img.shields.io/badge/STEP%201-Dataset_%26_Embeddings-blue)
**Goal:** Establish the baseline for dense retrieval.
*   **What the code does:** We load the **SciFact** benchmark (corpus and queries). We then use the `all-mpnet-base-v2` sentence transformer model to encode all texts into 768-dimensional dense vectors. 
*   **Why we do this:** We need a highly precise dataset (scientific claims) and a standard, robust embedding space (768 dimensions) to serve as our "Ground Truth" for full-resolution semantic retrieval.

---

### ![STEP 2](https://img.shields.io/badge/STEP%202-Initial_Goal_%26_Ground_Truth-orange)
**Goal:** Determine the minimum successful dimension for every query.
*   **What the code does:** We iteratively truncate the embeddings (768 → 512 → 256 → 128 → 64 → 32), re-normalize them, and compute the `Recall@10` for each query. We log the exact dimension where a query "fails" to retrieve the relevant document.
*   **Why we do this:** Our initial research question was: *Do all queries require the same embedding dimension?* We wanted to build a dataset where the target variable was the absolute `Required Dimension` (e.g., this query needs 128, that query needs 32).

---

### ![STEP 3](https://img.shields.io/badge/STEP%203-The_Pivot_(Why_we_switched)-red)
**Goal:** Realize the limitation of direct dimension prediction.
*   **What the code does (or fails to do):** Early attempts in the notebook to predict the exact `Required Dimension` using standard query features yielded extremely weak correlations. 
*   **Why we switched:** The `Required Dimension` is a discrete, step-based variable (it jumps from 64 to 128). It is highly unstable; a single irrelevant document jumping in rank can alter the "Required Dimension" drastically. We realized that predicting the absolute dimension was the wrong framing for an unsupervised analysis. We needed a continuous measure of *robustness*.

---

### ![STEP 4](https://img.shields.io/badge/STEP%204-Rank_Shift_Metric-brightgreen)
**Goal:** Introduce a stable measure of retrieval robustness.
*   **What the code does:** Instead of finding the drop-off point, the code tracks the exact rank of the *first relevant document* at 768 dimensions ($r_{768}$) and its new rank when truncated to 32 dimensions ($r_{32}$). It calculates the absolute difference: `Rank Shift = |r_32 - r_768|`.
*   **Why we do this:** `Rank Shift` is continuous. If a document drops from rank 1 to rank 5, it's a small shift (robust). If it drops from 1 to 500, it's a massive shift (fragile). This metric perfectly captures semantic degradation without the instability of discrete dimension steps.

---

### ![STEP 5](https://img.shields.io/badge/STEP%205-Feature_Extraction-purple)
**Goal:** Quantify query characteristics in the full-dimensional space.
*   **What the code does:** For every query, using the 768-dimensional scores, the code calculates:
    1.  **Similarity Margin:** The gap between the Top-1 and Top-2 documents.
    2.  **Entropy:** The distribution of the top scores.
    3.  **Variance:** The spread of the top scores.
    4.  **Query Length:** Number of words.
    5.  **Relevant Docs:** Total ground-truth documents.
*   **Why we do this:** We need to test the assumption that standard "Query Difficulty" metrics (like Length or Entropy) dictate how well a query compresses.

---

### ![STEP 6](https://img.shields.io/badge/STEP%206-Correlation_%26_Findings-gold)
**Goal:** Prove what actually determines dimensional robustness.
*   **What the code does:** Computes the Spearman correlation ($\rho$) between the features from STEP 5 and the `Rank Shift` from STEP 4. It then generates the scatter plots.
*   **The Findings:** 
    *   ✅ **Main Finding:** Margin is the strongest indicator ($\rho \approx -0.54, p < 10^{-23}$). A large initial gap means the query survives truncation.
    *   ✅ **Negative Result:** Entropy, Variance, and Query Length have weak or negligible correlations.
*   **Why this matters:** It proves that compression viability depends primarily on the local neighborhood topology (the Margin), not the global semantic complexity (like Query Length).

---

### ![STEP 7](https://img.shields.io/badge/STEP%207-Future_Work-darkgrey)
**Goal:** Set up the next phase of research.
*   **What the code does:** Contains experimental blocks (like Decision Trees or Regression) that attempt to predict the dimension directly.
*   **Why we leave this out of the paper:** For this conference paper, establishing the unsupervised *phenomenon* (Margin predicts Rank Shift) is a complete, novel contribution. Building the actual Machine Learning pipeline to route queries (Cascaded Retrieval) is left as future work, keeping the current paper highly focused and scientifically rigorous.
