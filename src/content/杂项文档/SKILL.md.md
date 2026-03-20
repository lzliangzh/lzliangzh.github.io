
# Skill: Autonomous Research Trajectory & Topic Synthesis

## 1. Objective

To autonomously navigate the $ArXiv$ and top-tier conference landscapes (NeurIPS, ICML, ICLR, CVPR, etc.) to identify "Research Gaps" and formulate high-probability-of-acceptance (PoA) research topics.

---

## 2. Phase I: Latent Keyword Expansion (The "Entry Point")

The agent must not rely on the user's initial prompt alone. It must perform **Semantic Expansion**.

- **Step 1: Seed Extraction.** Identify core concepts from the user's direction (e.g., "Efficient LLM Training").
    
- **Step 2: Taxonomy Mapping.** Query the LLM to generate:
    
    - **Direct Synonyms:** (e.g., "Parameter-efficient fine-tuning", "PEFT").
        
    - **Proximal Technologies:** (e.g., "Quantization", "Low-rank adaptation", "Sparse backpropagation").
        
    - **Problem-Space Keywords:** (e.g., "Memory constraints", "VRAM bottleneck", "Communication overhead").
        
- **Step 3: Recursive Refinement.** Search the first 10 results on ArXiv for the seed. Extract "Keywords" or "CCS Concepts" from these papers to update the search vector.
    

---

## 3. Phase II: Multi-Dimensional Quality Filtering

Finding "high-quality" papers requires looking beyond just the title.

|**Metric**|**Threshold / Logic**|
|---|---|
|**Venue Filter**|Cross-reference ArXiv IDs with DBLP or Semantic Scholar to prioritize papers published in CCF-A or Core A* venues.|
|**Citation Velocity**|Calculate $\frac{Citations}{Months \ Since \ Publication}$. High velocity indicates a "hot" sub-field.|
|**Affiliation Weight**|(Optional) Prioritize labs with a history of SOTA (e.g., FAIR, DeepMind, Stanford, Tsinghua).|
|**Code Availability**|Prioritize papers with linked GitHub repositories (higher feasibility for reproduction).|

---

## 4. Phase III: Systematic Literature Deconstruction

For the top $N$ papers (typically 5–10), the agent must extract a structured "Insight Matrix":

1. **The "Mechanism":** What is the specific mathematical or architectural contribution?
    
2. **The "Benchmark Gap":** Where does the model fail? (Look at the "Limitations" section and the bottom of the "Results" tables).
    
3. **The "Assumptions":** What constraints did the authors take for granted? (e.g., "Assumes IID data," "Requires 8xA100 GPUs").
    

---

## 5. Phase IV: Topic Formation (The "Innovation" Logic)

The agent generates topics by applying **Relational Operators** to the current research status:

- **Operator A: Cross-Pollination.** Apply a technique from Domain X to the problem in Domain Y (e.g., "Diffusion-based augmentation for Tabular Reinforcement Learning").
    
- **Operator B: Constraint Relaxation.** If current SOTA requires $O(n^2)$ memory, can we achieve $O(n \log n)$?
    
- **Operator C: Robustness Stress-Test.** If current SOTA works on Benchmark A, does it fail under Adversarial Shift B?
    

---

## 6. Phase V: Feasibility & Target Conference Scoring

Each generated topic must be scored (1–10) based on:

- **Innovation:** Does it solve a "Limitation" identified in Phase III?
    
- **Feasibility:** Do we have the compute/datasets (Zotero/Local check)?
    
- **Alignment:** Does this fit the "Call for Papers" (CfP) of the next major conference (e.g., if the deadline is NeurIPS, focus on theoretical depth; if CVPR, focus on visual SOTA).
    

---

## 7. Execution Loop (The "Skill" Workflow)

Markdown

```
### AGENT_INSTRUCTIONS:
1. START with {{research_direction}}.
2. EXECUTE 'Latent Keyword Expansion' to generate 5 specific search strings.
3. SCRAPE ArXiv and Semantic Scholar. 
4. RANK results using the 'Quality Filtering' matrix.
5. READ the 'Conclusion/Limitations' of the top 3 papers.
6. SYNTHESIZE 3 Research Topics using 'Relational Operators'.
7. EVALUATE each topic against the 'Feasibility/Innovation' rubric.
8. OUTPUT: A "Research Roadmap" document including:
    - Summary of SOTA
    - 3 Proposed Topics
    - Suggested Baseline Models to modify
```


使用 AI 对于 Arxiv html 进行总结。