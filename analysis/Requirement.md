# [CISC7022] Course Project: Advanced Research in Scalable AI & Big Data


### Group Members: at most 2 students per group

### 1. Project Objective
The objective of this project is to move beyond passive comprehension of research literature. Students are tasked with performing an **adversarial technical audit** of a seminal or cutting-edge work at the intersection of Big Data and AI. You will investigate the "fragility" of proposed systems—evaluating how architectural trade-offs, methodology and hardware constraints, and data distribution shifts impact the validity of the authors' claims under extreme scale.

---

### 2. Scope & Paper Selection
Students must select a research focus from one of the following two tracks:

* **Track A (The Deep Dive):** Select one "Anchor" paper from a premier venue (e.g., **NeurIPS, ICML, OSDI, SOSP, SIGMOD, or VLDB**) published within the last 36 months.
* **Track B (The Comparative Audit):** Select two competing papers that approach the same fundamental bottleneck (e.g., **Vector Indexing for LLMs** or **Distributed Training Optimization**) using diametrically opposed philosophies (e.g., *Software-defined* vs. *Hardware-accelerated*).

---

### 3. Deliverables & Technical Requirements
Your final report must be **6–8 pages**, formatted using the **ACM SIGMOD double-column template** (https://2026.sigmod.org/calls_papers_sigmod_research.shtml#submission). It should be written as a technical critique, not a summary.

#### Section I: Architectural Synthesis (20%)
* **The "Core Innovation" Calculus:** Succinctly define the technical delta. What specific mathematical or system-level insight differentiates this from the "prior state-of-the-art" (SOTA)?
* **Efficiency Frontier:** Map the paper's contribution onto the efficiency frontier. Does it optimize for **latency, throughput, energy efficiency**, or **model convergence rate**?

#### Section II: Technical Stress Test & System Scalability (60%)
*This section is the core of your report. You must bridge the gap between the paper’s theoretical "happy path" and the "messy reality" of Big Data production.* The following parts are optional, but they are strongly recommended if you want to produce a deeper and more critical report. These sections are intended to help you connect the paper’s idealized technical design with the practical realities of real-world large-scale systems.

* **Part A (Optional): System Reconstruction & Logic** : Reconstruct the paper’s main technical idea at a high level using pseudocode, a workflow diagram, or a structured step-by-step description. Focus on the logic of the method rather than implementation details. In particular, explain how data, computation, and intermediate results flow through the system. Discuss whether the design mainly moves computation to where the data resides, moves data to the compute resources, or uses some hybrid strategy, and comment on the likely implications of this design.

* **Part B (Optional): Operational Assumptions & Failure Conditions**: Identify one or two important assumptions or environmental conditions that the proposed method relies on in order to work well. These may involve data organization, hardware characteristics, workload balance, communication efficiency, memory availability, or input distribution assumptions. Then discuss possible failure conditions or "breaking points": under what circumstances would performance degrade sharply, become unstable, or no longer scale effectively? Where possible, give a rough quantitative estimate, reasoned threshold, or illustrative example.

* **Part C (Optional): Scalability & Sensitivity Analysis**: Analyze how the method responds when a key system or model parameter changes substantially, such as data size, number of workers, batch size, graph size, memory budget, or communication cost. You may use logical reasoning, a simplified numerical example, or a toy simulation. Where appropriate, students are encouraged to include simple implementations, controlled experiments, or empirical observations to support their analysis, although this is not compulsory.

#### Section III: Simulated Peer Review (SPC Perspective) (20%)
Assume you are a Senior Program Committee (SPC) member. Write a **"Weak Reject"** review (approx. 500 words) based on your findings in Section II, focusing on:
* **Hidden Costs:** Does the paper ignore the "Total Cost of Ownership" (e.g., specialized GPU requirements)?
* **Methodological Gaps:** Did the authors "cherry-pick" datasets or ignore standard industry baselines?
* **Practicality:** Is the system too complex to maintain in a production data pipeline?

---

### 4. Evaluation Rubric

| Criteria | Exceptional (A/A+) | Proficient (B) |
| :--- | :--- | :--- |
| **Critical Rigor** | Identifies non-obvious "hidden costs" and architectural fragility. | Understands the paper but accepts most author claims as valid. |
| **Technical Depth** | Provides formal complexity derivations, working "toy" proofs, detail analysis. | Explains the methods correctly but lacks independent derivation. |
| **Scalability Vision** | Maps behavior to time, space, or hardware constraints. | Discusses scaling in general terms without specific context. |

---

### 5. Integrity & AI Usage Policy
1.  **Mandatory Disclosure:** The main prompts used for code debugging or structural brainstorming must be logged in the "AI Usage Appendix." (excluding from the page limit)
2.  **The "Vocal Defense":** You will be required to defend your **Section II (Parts B and C)** in a 10-minute Q&A. The use of AI during the defense is strictly prohibited.
3.  **Hallucination Liability:** Any "hallucinated" citations or fabricated mathematical proofs will result in an immediate fail for the project.

---

### Recommended Reference Themes:
1.  **Vector Databases:** Optimization of HNSW vs. IVF-Flat for billion-scale embeddings.
2.  **Systems for LLMs:** Speculative Decoding or KV-Cache optimization in distributed inference.
3.  **Graph Learning:** Scalable GNN training on multi-GPU clusters.
4.  **Hardware-Aware AI:** Quantization-aware training (QAT) vs. Post-training quantization (PTQ).
