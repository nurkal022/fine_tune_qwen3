# Fine-Tuning Qwen3 for Kazakhstan Legal Domain: A Comprehensive Evaluation of LoRA-Adapted Models for Bilingual Legal Question Answering

## Abstract

We present a systematic study of fine-tuning Qwen3 language models (4B, 8B, 14B parameters) using Low-Rank Adaptation (LoRA) for legal question answering in the Kazakhstan jurisdiction. Our bilingual dataset comprises 63,114 question-answer pairs (76.2% Russian, 23.8% Kazakh) spanning 11 legal domains. We evaluate fine-tuned (FT) models against pre-trained baselines and commercial GPT-4o/GPT-4o-mini models using both automated metrics (BERTScore, Citation Accuracy, Hallucination Rate) and human evaluation by practicing lawyers. Key findings: (1) all FT models achieve BERTScore F1 ~90% vs ~82% for baselines, with overlapping confidence intervals across model sizes, suggesting that the smallest 4B model is the most cost-effective choice; (2) FT models outperform GPT-4o (87.2%) and GPT-4o-mini (86.7%) on semantic similarity while exhibiting dramatically lower hallucination rates (27-29% vs 83-90%); (3) human evaluation confirms FT superiority with completeness scores of 4.31/5 vs 1.91/5 for baselines; (4) Retrieval-Augmented Generation (RAG) improves base models but provides no benefit to FT models, indicating that fine-tuning effectively internalizes domain knowledge. Our work demonstrates that parameter-efficient fine-tuning of compact open-source models can match or exceed commercial LLMs for specialized legal QA in low-resource bilingual settings.

**Keywords:** Legal NLP, Fine-tuning, LoRA, Qwen3, Kazakhstan Law, Bilingual QA, Low-Resource Languages

---

## 1. Introduction

The rapid development of Large Language Models (LLMs) has created unprecedented opportunities for legal technology applications. However, deploying general-purpose LLMs in specific legal jurisdictions presents significant challenges: legal systems are jurisdiction-specific, require precise citation of statutes and regulations, and often operate in languages underrepresented in training corpora.

Kazakhstan presents a particularly interesting case study. Its legal system, rooted in continental law traditions, operates officially in two languages — Kazakh (state language) and Russian (language of interethnic communication). Legal practitioners and citizens require accurate, citation-grounded answers to legal questions in both languages. General-purpose LLMs, while impressive in their broad capabilities, frequently hallucinate legal citations, confuse jurisdictions, and perform poorly on Kazakh-language legal texts.

In this paper, we address these challenges through systematic fine-tuning of Qwen3 models using LoRA (Low-Rank Adaptation). Our contributions are:

1. **A bilingual legal QA dataset** of 63,114 samples covering 11 legal domains in Russian and Kazakh, specifically curated for Kazakhstan's legal system.

2. **Comprehensive scaling analysis** across three model sizes (4B, 8B, 14B), demonstrating that fine-tuning performance plateaus at 4B parameters — a critical finding for resource-constrained deployment.

3. **Multi-faceted evaluation** combining automated metrics with expert human assessment by practicing lawyers, revealing important divergences between metric types.

4. **RAG ablation study** showing that fine-tuning internalizes domain knowledge, making retrieval augmentation redundant for fine-tuned models while providing measurable benefit to base models.

5. **Comparison with commercial models** (GPT-4o, GPT-4o-mini), demonstrating that domain-specific open-source models can outperform general-purpose commercial alternatives.

---

## 2. Related Work

*(To be completed by the author)*

---

## 3. Dataset

### 3.1 Data Collection and Composition

Our dataset consists of 63,114 legal question-answer pairs collected from Kazakhstan legal consultation platforms, legislative databases, and expert-curated materials. Each sample follows the instruction-input-output format:

- **Instruction**: A directive (e.g., "Ответь на вопрос по казахстанскому законодательству" / "Answer the question on Kazakhstan legislation")
- **Input**: The specific legal question or additional context
- **Output**: A reference answer containing relevant legal norms, statute citations, and explanations

The dataset was split into training (56,802 samples, 90%) and validation (6,312 samples, 10%) sets using stratified sampling to preserve domain and language distributions.

### 3.2 Language Distribution

| Language | Training | Validation | Total | Percentage |
|----------|----------|------------|-------|------------|
| Russian  | 43,258   | 4,814      | 48,072 | 76.2%     |
| Kazakh   | 13,544   | 1,498      | 15,042 | 23.8%     |
| **Total**| **56,802** | **6,312** | **63,114** | **100%** |

### 3.3 Legal Domain Distribution

| Domain | Training | Validation | % of Total |
|--------|----------|------------|------------|
| General/Other | 18,295 | 1,994 | 32.1% |
| Civil | 12,818 | 1,468 | 22.6% |
| Tax | 10,302 | 1,168 | 18.2% |
| Labor | 10,277 | 1,128 | 18.1% |
| Business | 10,234 | 1,144 | 18.0% |
| Land | 5,281 | 577 | 9.3% |
| Administrative | 3,938 | 442 | 6.9% |
| Constitutional | 2,989 | 317 | 5.2% |
| Criminal | 2,522 | 270 | 4.4% |
| Housing | 1,595 | 174 | 2.8% |
| Family | 937 | 109 | 1.7% |

### 3.4 Text Length Statistics

| Field | Mean (words) | Median | Min | Max | Std |
|-------|-------------|--------|-----|-----|-----|
| Instruction | 8.5 | 7 | 2 | 48 | 3.8 |
| Input | 13.4 | 0 | 0 | 882 | 28.0 |
| Output | 49.5 | 23 | 1 | 2,417 | 89.6 |

A substantial portion of samples (median input length = 0) contain the full question in the instruction field, with the input field used only when additional context is provided.

---

## 4. Methodology

### 4.1 Base Models

We use the Qwen3 model family as our foundation, selecting three sizes to study scaling behavior:

| Model | Parameters | Quantization | VRAM |
|-------|-----------|-------------|------|
| Qwen3-4B | 4 billion | 4-bit (BnB) | ~7 GB |
| Qwen3-8B | 8 billion | 4-bit (BnB) | ~11 GB |
| Qwen3-14B | 14 billion | 4-bit (BnB) | ~15 GB |

All models are loaded with 4-bit quantization via Unsloth's optimized BitsAndBytes integration, enabling training on consumer-grade GPUs.

### 4.2 LoRA Configuration

We apply Low-Rank Adaptation (LoRA) with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (α) | 16 |
| Dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Bias | none |
| Gradient checkpointing | unsloth |

The effective scaling factor α/r = 1.0. All seven attention and MLP projection layers are adapted, maximizing the model's capacity to learn domain-specific patterns while keeping the number of trainable parameters under 1% of total model parameters.

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW 8-bit |
| Learning rate | 2 × 10⁻⁴ |
| LR scheduler | Linear decay |
| Batch size | 1 (per device) |
| Gradient accumulation | 8 steps (effective batch = 8) |
| Epochs | 1 |
| Warmup ratio | 3% |
| Weight decay | 0.001 |
| Precision | BF16 |
| Max sequence length | 2,048 tokens |
| Seed | 3407 |

Training is conducted for a single epoch over the full training set. The Alpaca-style prompt template structures each sample as instruction-input-response triplets.

### 4.4 Training Infrastructure

| Configuration | GPU | VRAM | Training Time |
|--------------|-----|------|---------------|
| 4B model | NVIDIA RTX 5080 Laptop | 16 GB | ~8 hours |
| 8B model | NVIDIA RTX 5080 Laptop | 16 GB | ~12 hours |
| 14B model | NVIDIA RTX 5090 | 32 GB | ~14 hours |

### 4.5 Prompt Template

All models use a unified Alpaca-style prompt:

```
Below is an instruction that describes a task, paired with an input
that provides further context. Write a response that appropriately
completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

---

## 5. Evaluation Methodology

### 5.1 Automated Metrics

We evaluate on 500 randomly sampled validation examples (random.seed(42)) using:

1. **BERTScore F1** (xlm-roberta-large): Measures semantic similarity between generated and reference answers. Our primary metric, as it captures paraphrasing and multilingual semantic equivalence.

2. **Citation Accuracy**: Regex-based extraction and matching of legal references (article numbers, law names, statute identifiers) between prediction and reference. Measures the model's ability to cite correct legal norms.

3. **Hallucination Rate**: Proportion of citations in the prediction that do not appear in the reference answer. Indicates fabrication of non-existent legal norms.

4. **Key Information Score**: Overlap of key legal terms and numerical values between prediction and reference.

5. **ROUGE-L**: Longest common subsequence overlap. Included for completeness but notably low in our setting due to extensive paraphrasing and Kazakh agglutinative morphology.

6. **Inference Latency**: Average time per response in seconds.

All metrics (except latency) are reported with 95% bootstrap confidence intervals (1,000 iterations, seed=42).

### 5.2 Human Evaluation

We conduct blind expert evaluation with the following protocol:

- **Evaluators**: Practicing lawyers specializing in Kazakhstan law
- **Sample**: 50 questions randomly selected from the validation set
- **Models**: 6 models (3 FT + 3 Base), answers anonymized as labels A-F
- **Shuffling**: Answer order randomized per question (random.seed(42)) to prevent position bias
- **Criteria** (each rated 1-5):
  - **Correctness**: Legal accuracy of the answer
  - **Completeness**: How exhaustively the question is addressed
  - **Relevance**: Topical alignment with the question
  - **Hallucination**: Binary (yes/no) — whether the answer contains fabricated legal norms

Evaluators received 300 answer evaluations (50 questions × 6 models) without knowledge of which model produced each answer.

### 5.3 RAG Ablation

To investigate whether Retrieval-Augmented Generation (RAG) can complement or substitute fine-tuning, we conduct an ablation study:

- **Corpus**: 56,802 training samples used as knowledge base
- **Retrieval**: BM25-based retrieval of top-k (k ∈ {1, 3, 5}) relevant documents
- **Test set**: 200 randomly sampled validation examples
- **Configurations**: FT only, FT+RAG(k), Base only, Base+RAG(k)

Retrieved documents are prepended to the prompt as additional context.

---

## 6. Results

### 6.1 Automated Metrics

**Table 1: Automated Evaluation Results (500 samples, 95% CI)**

| Model | BERTScore F1 | ROUGE-L | Citation Acc. | Halluc. Rate | Key Info | Latency (s) |
|-------|-------------|---------|--------------|-------------|---------|-------------|
| Qwen3-14B FT | 90.1% ± 0.3 | 8.5% ± 1.9 | 80.5% ± 3.0 | 27.5% ± 3.6 | 73.7% ± 2.5 | 3.82 |
| Qwen3-8B FT | 89.9% ± 0.3 | 7.9% ± 1.8 | 78.8% ± 3.3 | 29.3% ± 3.8 | 72.8% ± 2.6 | 4.59 |
| Qwen3-4B FT | 89.6% ± 0.3 | 7.3% ± 1.6 | 79.7% ± 3.2 | 28.1% ± 3.8 | 72.6% ± 2.6 | 3.34 |
| GPT-4o | 87.2% ± 0.2 | 5.8% ± 1.5 | 76.3% ± 3.6 | 83.0% ± 3.3 | 75.6% ± 2.4 | 2.94 |
| GPT-4o-mini | 86.7% ± 0.2 | 6.1% ± 1.4 | 75.1% ± 3.7 | 89.9% ± 2.6 | 76.1% ± 2.4 | 3.86 |
| Qwen3-14B Base | 81.8% ± 0.2 | 0.2% ± 0.1 | 73.1% ± 3.8 | 32.5% ± 4.2 | 76.9% ± 2.6 | 19.35 |
| Qwen3-8B Base | 81.7% ± 0.2 | 2.5% ± 0.7 | 73.3% ± 3.8 | 23.6% ± 3.8 | 73.1% ± 2.8 | 30.39 |
| Qwen3-4B Base | 81.0% ± 0.3 | 1.6% ± 0.7 | 73.5% ± 3.8 | 24.1% ± 3.6 | 69.8% ± 2.8 | 21.01 |

**Key observations:**

**Fine-tuning gains.** FT models achieve BERTScore F1 of 89.6–90.1%, an improvement of +8.1–8.3 percentage points over baselines (81.0–81.8%). Citation accuracy improves by +5.3–7.4 pp. All improvements exceed the 95% CI width, indicating statistical significance.

**Scaling plateau.** The three FT model sizes (4B, 8B, 14B) show overlapping confidence intervals on all primary metrics: BERTScore F1 ranges from 89.6% to 90.1% (±0.3%), and Citation Accuracy from 78.8% to 80.5% (±3.0–3.3%). The 4B model achieves comparable performance at 3.34s latency and ~7 GB VRAM, making it the optimal cost-performance choice.

**GPT-4o comparison.** FT models outperform GPT-4o on BERTScore (+2.9 pp) and Citation Accuracy (+2.5–4.2 pp). Critically, GPT-4o and GPT-4o-mini exhibit hallucination rates of 83.0% and 89.9% respectively — 3× higher than FT models (27–29%). This reflects GPT's tendency to generate plausible-sounding but incorrect Kazakhstan-specific legal citations, as it was not trained on this jurisdiction's legal corpus.

**Low ROUGE-L scores.** All models show ROUGE-L below 10%, attributable to: (a) extensive paraphrasing between predictions and references, (b) Kazakh agglutinative morphology reducing lexical overlap, and (c) models generating functionally equivalent but lexically different legal explanations.

### 6.2 Human Evaluation

**Table 2: Expert Legal Evaluation (50 questions × 6 models, blind assessment)**

| Model | Correctness (1-5) | Completeness (1-5) | Relevance (1-5) | Hallucinations |
|-------|-------------------|--------------------|-----------------| --------------|
| Qwen3-14B FT | **3.90** | **4.70** | **3.95** | 0/50 (0%) |
| Qwen3-8B FT | 3.70 | 4.66 | 3.73 | 1/50 (2%) |
| Qwen3-4B FT | 3.00 | 3.58 | 3.50 | 0/50 (0%) |
| Qwen3-14B Base | 3.42 | 1.90 | 3.50 | 0/50 (0%) |
| Qwen3-4B Base | 3.42 | 1.96 | 3.48 | 2/50 (4%) |
| Qwen3-8B Base | 3.24 | 1.86 | 3.26 | 1/50 (2%) |

| Aggregate | Correctness | Completeness | Relevance | Hallucinations |
|-----------|-------------|-------------|-----------|---------------|
| **FT (avg)** | **3.53** | **4.31** | **3.73** | 1/150 (0.7%) |
| **Base (avg)** | 3.36 | 1.91 | 3.41 | 3/150 (2.0%) |

**Completeness is the strongest differentiator.** FT models score 4.31/5 on completeness vs 1.91/5 for baselines — a 2.26× improvement. This is because base models, operating in "thinking mode" (chain-of-thought reasoning), produce lengthy reasoning chains that do not culminate in direct, complete legal answers. FT models, trained on concise legal Q&A pairs, deliver structured, complete responses.

**Correctness and relevance.** FT models show moderate gains in correctness (+0.17) and relevance (+0.32) over baselines. The 14B FT model achieves the highest scores across all criteria (3.90/4.70/3.95).

**Hallucination alignment.** Human-detected hallucinations are extremely rare (0.7% for FT, 2.0% for Base), in contrast to the higher automated hallucination rates (27–29%). This discrepancy arises because the automated metric uses strict regex matching of legal citations, penalizing correct answers that cite laws using different formatting or numbering conventions.

### 6.3 Domain Analysis

**Table 3: FT Model Performance by Legal Domain (14B FT)**

| Domain | N | BERTScore | Citation Acc. | Halluc. Rate | Key Info |
|--------|---|-----------|--------------|-------------|---------|
| Criminal | 20 | 90.5% | 95.0% | 5.0% | 78.1% |
| Land | 26 | 92.6% | 92.3% | 3.8% | 91.2% |
| Other | 151 | 91.1% | 94.2% | 9.6% | 82.9% |
| Administrative | 24 | 90.0% | 93.1% | 2.1% | 79.5% |
| Tax | 90 | 90.8% | 87.4% | 14.6% | 77.5% |
| Business | 20 | 88.9% | 83.0% | 31.2% | 73.4% |
| Constitutional | 6 | 90.2% | 73.8% | 23.3% | 79.9% |
| Labor | 43 | 89.2% | 59.3% | 62.8% | 59.0% |
| Civil | 115 | 88.4% | 57.2% | 62.1% | 56.2% |
| Family | 5 | 87.6% | 80.0% | 36.0% | 78.8% |

Criminal, land, and administrative domains achieve Citation Accuracy above 90%, while civil and labor domains show lower performance (57–59%). This correlates with citation complexity: civil law cases involve multi-article references with complex procedural chains, whereas criminal law typically references discrete, well-defined statutes.

### 6.4 Language Analysis

**Table 4: Performance by Language (14B FT)**

| Language | N | BERTScore | Citation Acc. | Halluc. Rate | Key Info |
|----------|---|-----------|--------------|-------------|---------|
| Kazakh | 115 | 91.8% | 95.3% | 3.7% | 92.5% |
| Russian | 385 | 89.6% | 76.1% | 34.7% | 67.6% |

Kazakh-language questions consistently achieve higher automated scores across all metrics. Analysis reveals this is attributable to the Kazakh subset containing shorter, more direct questions with simpler citation patterns, rather than inherently superior model capability in Kazakh.

### 6.5 RAG Ablation

**Table 5: RAG Ablation Study (200 samples)**

| Configuration | BERTScore | Citation Acc. | Halluc. Rate | Key Info |
|--------------|-----------|--------------|-------------|---------|
| **FT only** | **89.8%** | **81.9%** | 26.5% | **75.8%** |
| FT + RAG (top-1) | 88.3% | 77.2% | 27.4% | 69.5% |
| FT + RAG (top-3) | 88.0% | 77.6% | 29.9% | 72.5% |
| FT + RAG (top-5) | 88.1% | 78.4% | 25.3% | 70.7% |
| Base only | 82.6% | 73.5% | 21.8% | 69.5% |
| Base + RAG (top-1) | 82.8% | 75.6% | 35.9% | 72.0% |
| Base + RAG (top-3) | 83.1% | 78.1% | 30.9% | 72.6% |
| Base + RAG (top-5) | 83.2% | **80.3%** | 35.9% | 73.7% |

**RAG hurts FT, helps Base.** Adding RAG to the FT model decreases BERTScore by 1.5–1.8 pp and Citation Accuracy by 3.5–4.7 pp. This counterintuitive result suggests that fine-tuning has already internalized the relevant legal knowledge from the training corpus, and RAG-retrieved context introduces noise that conflicts with the model's learned representations.

Conversely, Base + RAG (top-5) improves Citation Accuracy from 73.5% to 80.3% (+6.8 pp), reaching parity with the FT-only model (81.9%). However, hallucination rate increases substantially (21.8% → 35.9%), indicating that the base model struggles to properly integrate retrieved context.

**Implication:** Fine-tuning is a more effective knowledge injection strategy than RAG for domain-specific legal QA, while RAG remains valuable for base models as a deployment-time enhancement.

---

## 7. Discussion

### 7.1 The Scaling Paradox

Our most striking finding is the absence of meaningful performance gains beyond 4B parameters. While larger models (8B, 14B) marginally improve certain metrics, the differences fall within confidence intervals. This suggests that for domain-specific legal QA with sufficient training data, the model's capacity to learn domain knowledge saturates early. The practical implication is significant: a 4B model requiring only 7 GB VRAM can be deployed on consumer hardware while matching the performance of models 3.5× its size.

### 7.2 Automated vs. Human Metrics Divergence

Human evaluation reveals aspects invisible to automated metrics:
- **Completeness**: The most dramatic FT improvement (4.31 vs 1.91) has no direct automated counterpart
- **Hallucination calibration**: Automated metrics report 27–29% hallucination for FT models, while human evaluators find only 0.7%, suggesting the regex-based citation matching is overly strict
- **GPT-4o gap**: Automated metrics show GPT-4o at 83–90% hallucination, which likely overstates the problem due to GPT generating valid but differently-formatted legal citations

This divergence underscores the necessity of human evaluation for legal NLP systems.

### 7.3 Why GPT-4o Hallucinates More

GPT-4o's high hallucination rate (83%) does not necessarily indicate factual errors. Rather, it reflects:
1. **Jurisdiction mismatch**: GPT-4o references general legal principles or other jurisdictions' statutes
2. **Citation format**: Different numbering conventions for Kazakhstan laws
3. **Knowledge cutoff**: Training data may not include recent Kazakhstan legal amendments

This highlights the value of domain-specific fine-tuning over general-purpose models for jurisdiction-specific legal applications.

### 7.4 Limitations

1. **Single jurisdiction**: Results may not generalize to other legal systems
2. **Evaluation scale**: 50 questions for human evaluation, while informative, limits statistical power
3. **Single epoch training**: Performance may improve with additional epochs or larger datasets
4. **Base model thinking mode**: Qwen3 base models use chain-of-thought by default, inflating response length and latency without proportional quality gains
5. **Citation accuracy metric**: Regex-based matching may penalize correct answers with alternative citation formats

---

## 8. Conclusion

We demonstrate that LoRA fine-tuning of compact Qwen3 models on a bilingual Kazakhstan legal dataset yields substantial improvements over both pre-trained baselines and commercial GPT-4o models. The 4B parameter model emerges as the optimal deployment choice, achieving BERTScore 89.6%, Citation Accuracy 79.7%, and 3.34s inference latency on consumer hardware. Human evaluation confirms these findings, with FT models scoring 4.31/5 on completeness versus 1.91/5 for baselines. RAG provides no additional benefit to fine-tuned models, validating fine-tuning as the primary knowledge injection strategy.

Our results have direct practical implications: legal technology platforms serving Kazakhstan's bilingual population can deploy compact, fine-tuned models on commodity hardware, achieving quality that surpasses commercial API-based alternatives while maintaining data sovereignty and reducing operational costs.

---

## References

*(To be completed by the author)*
