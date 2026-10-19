# Technical Report: Retrieval-Augmented Generation (Baseline vs. Cross-Encoder Re-Ranking)

**Author:** Sparsh Bhartia  
**GitHub Repository:** [https://github.com/Sparsh181/RAG-Baseline-vs-ReRanking](https://github.com/Sparsh181/RAG-Baseline-vs-ReRanking)  
**Course / Assignment:** RAG Assignment: Baseline vs. Re-Ranking  
**Document Selected:** *Dungeons & Dragons 5th Edition System Reference Document (SRD 5.1)*  
**Document Statistics:** 403 Pages | 245,138 Words | 1,094 Chunks (~400 tokens/chunk, 50 token overlap)  
**Models Used:**  
- **Bi-Encoder (Dense Embeddings):** `all-MiniLM-L6-v2` (384-dimensional dense vectors)  
- **Cross-Encoder (Re-Ranker):** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
- **Vector Index:** FAISS `IndexFlatIP` (Exact cosine similarity via L2-normalized vectors)  
- **LLM Generator:** Google Gemini (`gemini-2.0-flash` with fallback to deterministic grounded reader, `temperature = 0.0`)  
- **Hardware Evaluated:** Intel x86_64 CPU  

---

## 1. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

In this assignment, I implemented and compared two different retrieval strategies over the same chunked D&D rulebook: a baseline **Bi-Encoder** and a two-stage pipeline using a **Cross-Encoder** re-ranker.

| Dimension | Bi-Encoder (`all-MiniLM-L6-v2`) | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| :--- | :--- | :--- |
| **Input Structure** | Encodes query $q$ and document $d$ separately: $\mathbf{u} = f(q)$, $\mathbf{v} = f(d)$ | Concatenates query and document together: `[CLS] q [SEP] d [SEP]` |
| **Attention Mechanism** | Self-attention happens only inside $q$ and inside $d$; no cross-talk between them | Full cross-attention: every token in $q$ can attend to every token in $d$ across all layers |
| **Similarity Metric** | Simple dot product / cosine similarity: $\text{score} = \mathbf{u} \cdot \mathbf{v}$ | A classifier head scores the joint representation from the `[CLS]` token |
| **Offline Precomputation** | **Yes:** I can embed all 1,094 chunks once offline and store them in FAISS | **No:** I cannot precompute anything because the score depends on the exact $(q, d)$ pair |
| **Search Speed** | Milliseconds ($O(\text{len}(q)^2)$ to embed query + fast vector lookup) | Slower ($O((\text{len}(q) + \text{len}(d))^2)$ for every candidate pair at runtime) |
| **Understanding Quality** | Lower: Squeezes an entire chunk into a single 384-number vector | Higher: Reads words together to catch exact conditions, negations, and phrasing |

### How I Understand the Core Trade-Offs

1. **The Vector Bottleneck:**  
   With the bi-encoder, I am compressing a 400-token chunk into just 384 numbers. While this works well for broad topic matching, subtle nuances get washed out. If a passage contains a tiny qualifier like *"except when"*, *"not allowed"*, or a specific number, that detail gets diluted across the 384 numbers.
2. **Joint Token Interaction:**  
   The cross-encoder feeds the query and chunk into the transformer side-by-side. If I ask *"Can a rogue use sneak attack with a greatsword?"*, the model can directly compare the word *"greatsword"* against the requirement clause *"must use a finesse or ranged weapon"*. It immediately notices that a greatsword doesn't qualify.
3. **The Practical Compromise:**  
   The bi-encoder gives me raw speed because I can search through thousands of indexed chunks in a few milliseconds. The cross-encoder gives me deep reading comprehension, but it is too slow to run on everything. Combining them—using the bi-encoder to pick the top 10 candidates and the cross-encoder to pick the best 3—gives me the best of both worlds.

---

## 2. Qualitative & Rank Shift Analysis (Failure Mode Hunting)

To test the real-world benefit of re-ranking, I designed four stress-test queries based on common retrieval failure patterns: a **Keyword Trap**, an **Exception / Corner Case**, a **Negation / Constraint**, and a **Buried Chunk**. 

In all four cases, the baseline bi-encoder failed to place the correct answering chunk in the Top 3 (causing the LLM to hallucinate or miss key rules), while the cross-encoder successfully promoted the correct chunk into the Top 3.

### Rank Shift Summary Table

| Category | Query | Gold Chunk | Page | Baseline Rank | Re-Ranked Rank | Rank Shift | Answer 1 (Baseline) Verdict | Answer 2 (Re-Ranked) Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Keyword Trap** | *Can a rogue use sneak attack with a heavy two-handed greatsword?* | **#157** / **#87** | 66 / 39 | **#4** / **#6** | **#1** / **#2** | **#4 $\rightarrow$ #1**<br>**#6 $\rightarrow$ #2** | **Omission / Hallucination:** Baseline matched weapon stats chunks; missed the finesse requirement. Claimed sneak attack works with advantage. | **Correct & Complete:** Promoted the sneak attack rule (#87) and weapon table (#157); clearly stated greatswords cannot sneak attack. |
| **Exception / Corner Case** | *What is the minimum Constitution saving throw DC to maintain concentration when taking damage?* | **#988** | 357 | **#9** | **#2** | **#9 $\rightarrow$ #2** | **Omission:** Retrieved general saving throw chunks; completely missed the minimum DC 10 floor rule. | **Correct & Complete:** Promoted the exact concentration chunk; gave the full rule (DC 10 or half damage, whichever is higher). |
| **Negation & Constraints** | *What actions or reactions can a creature take while under the Incapacitated condition?* | **#993** | 359 | **#4** | **#1** | **#4 $\rightarrow$ #1** | **Omission:** Matched generic rules about taking combat actions; missed the condition rule prohibiting actions. | **Correct & Complete:** Promoted the condition definition; correctly stated no actions or reactions are allowed. |
| **Buried Chunk** | *What specific rules and attack penalties apply to melee attacks while fighting underwater?* | **#239** | 95 | **#7** | **#2** | **#7 $\rightarrow$ #2** | **Omission:** Chunks about normal 5-foot melee reach took top spots; missed the underwater disadvantage penalty. | **Correct & Complete:** Promoted underwater combat rules; correctly identified the disadvantage penalty and weapon exceptions. |

### Detailed Failure Walkthroughs

#### 1. Keyword Trap / Distractor
- **My Query:** *"Can a rogue use sneak attack with a heavy two-handed greatsword?"*
- **What went wrong in the baseline:** The bi-encoder heavily matched words like *"greatsword"*, *"two-handed"*, *"heavy"*, and *"attack"*. As a result, ranks 1 to 3 were filled with generic weapon tables and descriptions of two-handed weapons. The actual rule—which explicitly states that sneak attack *"must use a finesse or a ranged weapon"* (Page 39)—was buried at **Rank #6**.
- **Impact on the LLM:** Without Page 39 in the prompt context, Answer 1 didn't know about the finesse requirement and incorrectly stated that the rogue could sneak attack as long as they had advantage.
- **How the cross-encoder fixed it:** The cross-encoder evaluated the whole relationship between *"rogue"*, *"sneak attack"*, and *"greatsword"*. It promoted Chunk #157 (weapons table) to **Rank #1** and Chunk #87 (rogue class features) to **Rank #2**. With both chunks available, Answer 2 gave the correct answer: *"No, sneak attack requires a finesse or ranged weapon, and a greatsword has neither property."*

#### 2. Negation & Constraints
- **My Query:** *"What actions or reactions can a creature take while under the Incapacitated condition?"*
- **What went wrong in the baseline:** Dense vector search grabbed chunks discussing how actions and reactions work in normal combat turns. The actual condition text stating *"An incapacitated creature can't take actions or reactions"* (Page 359) was pushed down to **Rank #4**.
- **How the cross-encoder fixed it:** The cross-encoder understood the negative constraint in *"under the Incapacitated condition"* and boosted the condition chunk from **Rank #4 $\rightarrow$ Rank #1**, enabling the LLM to give a clear, accurate answer.

#### 3. An Honest Limitation I Discovered
- **Observation:** Re-ranking can only work with what the first stage gives it. In my pipeline, I set the first-stage pool to $K = 10$. If an answer requires multiple chunks across different chapters and one of those chunks is ranked at **Rank #14** by the bi-encoder, the cross-encoder will never see it. Re-ranking cannot replace good initial recall; it can only fix the ordering of the candidate pool.

---

## 3. Latency Profiling

I measured the latency of each component on my Intel x86_64 CPU across 10 repeated runs after a warm-up query.

### Latency Measurement Table

| Pipeline Component | Mean Latency | Std Dev | What It Does |
| :--- | :---: | :---: | :--- |
| **Query Embedding + FAISS Search** | **11.36 ms** | $\pm$ 0.92 ms | Encodes query with bi-encoder and searches FAISS vector index |
| **Cross-Encoder Scoring (10 pairs)** | **651.58 ms** | $\pm$ 26.41 ms | Runs joint cross-attention over all 10 candidate (query, chunk) pairs |
| **Total Re-Ranking Retrieval (Bi + CE)** | **662.94 ms** | $\pm$ 26.26 ms | Full two-stage candidate retrieval and re-ranking time |
| **LLM Generation (Deterministic Grounded)** | **0.02 ms** | $\pm$ 0.00 ms | Grounded extractive reader (online Gemini API adds ~800–1,200 ms) |
| **Pipeline 1 Total (Baseline)** | **11.38 ms** | $\pm$ 0.92 ms | Baseline retrieval + local prompt generation |
| **Pipeline 2 Total (Re-Ranked)** | **662.95 ms** | $\pm$ 26.26 ms | Two-stage retrieval + re-ranking + local prompt generation |

### My Takeaways on Latency

1. **The Overhead Factor:** Scoring 10 candidate pairs with the cross-encoder on CPU added about **650 ms**. That makes the cross-encoder about **$58\times$ slower** than initial vector search alone.
2. **Is this acceptable in practice?** Yes. In a production app using an online LLM (like Gemini or GPT-4), LLM generation takes 800 ms to 2,500 ms anyway. On a GPU, cross-encoding 10 pairs takes only **5 to 10 ms**. Adding 10 ms to a 1.5-second total response time is unnoticeable to a user, but it completely eliminates critical omissions and hallucinations.
3. **Hardware Considerations:** While running on CPU took ~65 ms per candidate pair, running on a GPU or using quantized models (like INT8 with ONNX Runtime) brings this down to under **8 ms** for all 10 pairs combined.

---

## 4. Scaling Analysis: Why Not Cross-Encode Everything?

**Question:** *Why don't I just run the Cross-Encoder over the entire document from the start and skip the Bi-Encoder completely?*

### 1. Mathematical Scaling ($O(\log N)$ vs. $O(N)$)
- Let $N$ be the number of chunks in the corpus.
- **Bi-Encoder + Vector Index:**  
  I encode all $N$ chunks **once offline**. When a user asks a question, I only need to run the transformer once on the query, and then do a vector similarity search. Vector search scales logarithmically ($O(\log N)$ with indexes like HNSW) or as a fast linear scan. The expensive transformer forward pass is never repeated for the documents.
- **Pure Cross-Encoder:**  
  Because the cross-encoder takes `(query, chunk)` together, I cannot precompute document embeddings. Every time a question is asked, I must run the full transformer model **$N$ separate times**—once for every chunk in the collection ($O(N)$ complexity).

### 2. Concrete Scale Projections
To see what this means in practice, I projected the runtimes from my test setup to larger collections:

| Corpus Size ($N$) | Bi-Encoder Search Time | Pure Cross-Encoder (CPU: 65 ms/pair) | Pure Cross-Encoder (GPU: 0.8 ms/pair) |
| :--- | :---: | :---: | :---: |
| **1,094 chunks** (My D&D Document) | **~11 ms** | **~71 seconds** (1.2 min) | **~875 ms** |
| **50,000 chunks** (Small Company Knowledge Base) | **~18 ms** | **~54 minutes** | **~40 seconds** |
| **1,000,000 chunks** (Large Document Library) | **~35 ms** | **~18 hours** | **~13.3 minutes** |

### 3. Conclusion
Trying to cross-encode an entire corpus at search time is completely impractical for real-world applications. Waiting 71 seconds for a single question on a 400-page book is unusable, and on a million chunks it would take hours.

That is why the **two-stage architecture** is so effective:
1. **Stage 1 (Bi-Encoder):** Uses precomputed vectors to quickly narrow down the entire corpus from 1,000,000 chunks to just 10 candidates in a few milliseconds.
2. **Stage 2 (Cross-Encoder):** Spends its computational budget carefully, running deep joint attention only on those 10 candidates to pick the top 3.
