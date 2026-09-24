# Evaluating Dense, Hybrid, and Threshold-Based Retrieval Strategies for Enterprise RAG Systems

## Abstract

Retrieval-Augmented Generation (RAG) systems depend on retrieval quality to provide language models with evidence that is both relevant to a user's query and sufficient to support an answer. In enterprise settings, retrieval introduces additional challenges because questions may depend on multiple document sections, contain organization-specific terminology, or fall outside the knowledge base entirely. This study investigates how dense retrieval, similarity-based evidence thresholds, and hybrid dense-sparse retrieval affect retrieval accuracy, unsupported-query rejection, multi-section evidence coverage, and downstream answer quality.

We implement and evaluate an enterprise-style RAG system using dense semantic retrieval, BM25 sparse retrieval, reciprocal rank fusion, similarity-based evidence selection, and grounded language-model generation. The evaluation uses a controlled corpus of 20 synthetic enterprise documents containing approximately 100 indexed chunks. The retrieval benchmark contains 150 question cases comprising known-answer, unsupported, and multi-section queries. A separate 60-case benchmark evaluates downstream answer correctness, faithfulness, citation accuracy, and abstention behavior.

The study is designed to examine a central retrieval trade-off: stricter similarity filtering can reduce unsupported evidence while simultaneously excluding individually lower-scoring passages that are collectively necessary to answer multi-section questions. Hybrid retrieval provides a complementary lexical signal that can recover evidence containing domain-specific terminology that may not achieve a sufficiently high dense similarity score. The experiments therefore evaluate retrieval not only as a ranking problem, but also as an evidence-coverage problem for grounded generation.

The results show that the unfiltered hybrid configuration achieved 100.00% Top-1 accuracy and Recall@3 on supported retrieval cases, while threshold-based evidence selection increased unsupported-query rejection to 93.33% and overall benchmark accuracy to 98.00%. Downstream answer evaluation achieved 94.44% correctness, 93.33% faithfulness, and 93.33% citation accuracy, while fully successful multi-section answers reached 80.00%. The findings characterize the trade-off between unsupported-query rejection and evidence coverage and provide a reproducible basis for further research into retrieval and evidence selection for enterprise RAG systems.

**Keywords:** Retrieval-Augmented Generation, RAG, enterprise AI, dense retrieval, sparse retrieval, BM25, hybrid retrieval, reciprocal rank fusion, information retrieval, large language models, retrieval evaluation, evidence coverage

---

# 1. Introduction

Large language models (LLMs) have demonstrated strong capabilities across a broad range of language understanding and generation tasks. However, relying exclusively on knowledge encoded in model parameters creates practical limitations for applications that require access to private, domain-specific, frequently changing, or explicitly traceable information. Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving external information and providing the retrieved evidence to a generative model at inference time.

The original RAG formulation combined parametric language-model memory with non-parametric external memory accessed through retrieval, demonstrating the value of retrieval for knowledge-intensive language tasks [1]. Subsequent work on dense passage retrieval showed that learned vector representations could provide highly effective semantic retrieval for open-domain question answering [2]. Traditional sparse retrieval approaches such as BM25 remain important because lexical matching can capture exact terminology and query-document relationships that are not necessarily represented by semantic similarity alone [3]. Reciprocal Rank Fusion (RRF) provides a simple mechanism for combining ranked retrieval results from different retrieval systems [4].

These techniques have become common building blocks for modern RAG systems. However, enterprise RAG introduces an important practical problem that is less visible when retrieval is evaluated only as a ranking task.

A retrieved passage can be semantically related to a question without containing sufficient evidence to answer it.

This distinction becomes particularly important when a system applies a similarity threshold to determine whether retrieved evidence should be passed to the generation model. A threshold can serve as a useful mechanism for rejecting unsupported questions and reducing irrelevant context. At the same time, a globally applied threshold may remove passages that individually have moderate similarity but collectively contain the evidence required to answer a multi-part or multi-section question.

Consider an enterprise question such as:

> "What happens when a vendor is onboarded and later offboarded?"

The answer may require evidence from two distinct document sections: one describing vendor onboarding and another describing vendor offboarding. A retrieval system that evaluates each passage independently may assign neither passage a sufficiently high similarity score to survive an aggressive threshold, even though the two passages together provide the required evidence.

This creates a retrieval trade-off between **precision of individual evidence selection** and **coverage of the evidence required to answer the question**.

The problem is particularly relevant to enterprise systems because enterprise knowledge bases frequently contain structured policies, procedures, operational guidelines, security documentation, human-resources policies, vendor processes, and other documents in which related concepts are distributed across multiple sections. Enterprise questions can therefore require the retrieval system to identify several complementary pieces of evidence rather than a single highly similar passage.

This study investigates that trade-off experimentally.

We construct an enterprise-style RAG system and evaluate several retrieval configurations:

1. Dense semantic retrieval.
2. Dense retrieval with similarity-based evidence filtering.
3. Hybrid dense-sparse retrieval.
4. Hybrid retrieval combined with evidence selection.
5. The effect of retrieval configuration on downstream answer quality.

The evaluation explicitly includes three categories of questions:

- **Known questions**, for which the required evidence exists in the corpus.
- **Unknown questions**, for which the corpus does not contain sufficient information.
- **Multi-section questions**, for which the answer requires evidence from multiple document sections.

This design allows retrieval quality to be evaluated from more than one perspective. A system that retrieves relevant passages for known questions but also retrieves plausible-looking evidence for unsupported questions may create undesirable downstream behavior. Conversely, a system that aggressively rejects unsupported queries but removes legitimate evidence from multi-section questions may also reduce answer quality.

The study therefore treats unsupported-query rejection and evidence coverage as complementary evaluation dimensions rather than optimizing only for a single retrieval accuracy measure.

## 1.1 Research Problem

The central problem investigated in this work is:

> **How should evidence selection in an enterprise RAG system balance semantic relevance, unsupported-query rejection, and coverage of the multiple pieces of evidence required to answer a question?**

Traditional retrieval metrics primarily evaluate whether relevant documents appear near the top of a ranked list. While this remains important, downstream RAG systems introduce an additional requirement: the retrieved context must contain enough supporting evidence for the generator to produce a grounded answer.

This distinction motivates the experimental comparison between dense retrieval, threshold-based filtering, and hybrid retrieval.

## 1.2 Research Questions

The study investigates the following research questions.

### RQ1 — Dense Retrieval

How accurately does dense semantic retrieval identify the relevant evidence for known enterprise questions?

### RQ2 — Similarity Thresholds

How does applying a dense similarity threshold affect known-query retrieval accuracy and rejection of unsupported questions?

### RQ3 — Evidence Coverage

How do similarity thresholds affect questions that require evidence from multiple document sections?

### RQ4 — Hybrid Retrieval

Can combining dense semantic retrieval with sparse lexical retrieval improve evidence coverage for enterprise questions?

### RQ5 — Downstream Answer Quality

How do differences in retrieval and evidence selection affect downstream answer correctness, faithfulness, citation accuracy, and abstention behavior?

### RQ6 — Operational Trade-offs

What operational charecteristics, including latency, context size, and estimated generation cost, are observed across the evaluated RAG pipeline?

---

# 2. Research Contributions

This study makes the following contributions.

### 2.1 Controlled Enterprise RAG Evaluation Framework

We develop a reproducible evaluation framework for examining retrieval behavior in an enterprise-style RAG system. The framework separates retrieval evaluation from downstream answer evaluation and explicitly represents known, unsupported, and multi-section questions.

### 2.2 Evidence-Coverage Perspective

We investigate retrieval as an evidence-coverage problem in addition to a ranking problem. In particular, we examine cases where individually moderate-scoring passages are collectively necessary to answer a question.

### 2.3 Threshold Trade-off Analysis

We empirically evaluate the effect of similarity-based evidence filtering on both supported and unsupported questions. This allows the study to characterize the trade-off between rejecting unsupported evidence and accidentally removing valid evidence.

### 2.4 Dense-Sparse Retrieval Comparison

We compare dense semantic retrieval with a hybrid dense-sparse approach using BM25 and reciprocal rank fusion. The goal is not to introduce a new retrieval algorithm, but to investigate how complementary semantic and lexical signals affect evidence coverage in enterprise-style RAG.

### 2.5 Retrieval-to-Generation Evaluation

We connect retrieval behavior to downstream answer quality through an independent answer-quality benchmark measuring correctness, faithfulness, citation accuracy, and abstention behavior.

### 2.6 Reproducible Research Artifacts

The implementation, evaluation cases, benchmark configuration, result artifacts, and research documentation are maintained as part of the project repository to support reproducibility and future extensions.

---

# 3. Hypotheses

Based on the research questions, we formulate the following hypotheses.

### H1 — Dense Retrieval Effectiveness

Dense semantic retrieval will provide high retrieval accuracy for supported enterprise questions because semantic representations can identify conceptually related passages even when exact query terms differ.

### H2 — Threshold Trade-off

Increasing the similarity threshold will improve rejection of unsupported questions but may reduce evidence coverage for supported questions, particularly when the required evidence is distributed across multiple passages.

### H3 — Lexical Complementarity

Combining dense retrieval with sparse lexical retrieval will provide complementary evidence for queries containing domain-specific terminology and exact operational concepts.

### H4 — Multi-Section Coverage

Hybrid retrieval will provide more robust evidence coverage for questions whose answers require information from multiple document sections.

### H5 — Downstream Impact

Retrieval configurations that provide more complete supporting evidence will generally produce better downstream answer correctness, faithfulness, and citation behavior than configurations that provide incomplete evidence.

These hypotheses are evaluated empirically rather than assumed to hold.

---

# 4. Related Work

## 4.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation was introduced as a framework that combines parametric language-model memory with non-parametric memory accessed through retrieval [1]. The approach demonstrated that retrieved external knowledge can improve performance on knowledge-intensive tasks while providing a mechanism for incorporating information outside the model's parameters.

Modern enterprise RAG systems extend this general architecture by retrieving organization-specific documents and using them as context for generation. This introduces practical retrieval questions involving document segmentation, retrieval ranking, filtering, evidence selection, and grounding.

The present study focuses specifically on the interaction between retrieval filtering and evidence coverage rather than proposing a new generative architecture.

## 4.2 Dense Retrieval

Dense retrieval represents queries and passages using continuous vector representations and ranks candidate passages according to vector similarity. Dense Passage Retrieval demonstrated that a dual-encoder architecture could provide highly effective passage retrieval for open-domain question answering [2].

Dense retrieval is particularly attractive for enterprise RAG because semantic representations can retrieve conceptually related information even when query wording differs from the source documentation.

However, semantic similarity does not necessarily guarantee that a passage contains all of the evidence required to answer a question. This motivates the examination of dense retrieval scores as evidence-selection signals rather than treating them as direct measures of answer sufficiency.

## 4.3 Sparse Retrieval and BM25

Sparse retrieval methods use lexical statistics to estimate the relevance of documents to a query. BM25 is one of the most established probabilistic retrieval approaches and remains widely used in information-retrieval systems [3].

Lexical retrieval provides a complementary signal to dense semantic retrieval. Exact terminology can be particularly useful when enterprise documents contain specialized names, policy terms, product names, process labels, or other vocabulary for which lexical overlap provides strong evidence of relevance.

This study therefore evaluates BM25 as a complementary retrieval mechanism rather than as a replacement for dense retrieval.

## 4.4 Reciprocal Rank Fusion

Reciprocal Rank Fusion combines ranked lists from multiple retrieval systems by assigning higher weight to documents appearing near the top of each ranking [4]. Its simplicity makes it suitable for combining retrieval systems with different scoring mechanisms.

In this study, RRF is used to combine dense and sparse rankings. The purpose is to investigate whether combining semantic and lexical rankings can improve evidence coverage in enterprise-style RAG questions.

## 4.5 Retrieval Evaluation in RAG

Retrieval quality is a critical component of RAG because generation quality is constrained by the evidence supplied to the language model. A system may produce fluent responses while still failing to answer correctly when relevant evidence is missing or irrelevant evidence is retrieved.

This motivates evaluating retrieval independently from generation and then examining the relationship between retrieval behavior and downstream answer quality.

The present study extends this perspective by explicitly evaluating unsupported queries and multi-section questions. These categories allow the evaluation to distinguish between retrieving plausible evidence and retrieving sufficient evidence.

The study therefore focuses on an evidence-selection trade-off that is particularly relevant to enterprise RAG: improving rejection of unsupported queries while preserving the multiple pieces of evidence required for spported questions.

## 4.6 Recent RAG Evaluation Framework

Recent work has increasingly treated RAG evaluation as a multi-dimensional problem rather than evaluating only final answer accuracy. RAGAs introduced a reference-free evaluation framework covering multiple aspects of retrieval and generation, including the relevance of retrieved context and the faithfulness of generated responses [5]. This supports the separation of retrieval quality from downstream generation quality adopted in the present study.

ARES further formalized automated evaluation of RAG systems through separate measurements of context relevance, answer faithfulness, and answer relevance [6]. ARES also uses a small amount of human-annotated data with prediction-powered inference to improve the reliability of automated evaluation. The present study similarly separates retrieval and generation evaluation, but uses explicit benchmark targets for supported and unsupported questions and evaluates multi-section evidence coverage.

RAGChecker proposed a fine-grained diagnostic framework for evaluating both retrieval and generation behavior and reported that its metrics correlated more strongly with human judgments than several alternative evaluation approaches [7]. This motivates the present study's emphasis on error categories such as unsupported-query rejection, evidence coverage, and downstream faithfulness rather than relying on a single aggregate answer score.

Benchmarking work such as RGB has also identified negative rejection and information integration as important capabilities for RAG systems [8]. These dimensions are closely related to the unsupported-query and multi-section cases evaluated in this study.

The present work differs in scope by focusing specifically on the interaction between similarity-based evidence selection, hybrid retrieval, unsupported-query rejection, and multi-section evidence coverage within a controlled enterprise-style corpus.

---

# 5. Methodology

## 5.1 System Architecture

The experimental system follows the following pipeline:

    Document
        |
        v
    Heading-aware chunking
        |
        v
    Embedding generation
        |
        v
    Vector index
        |
        +----------------------+
        |                      |
        v                      v
    Dense retrieval       BM25 retrieval
        |                      |
        +----------+-----------+
                   |
                   v
          Reciprocal Rank Fusion
                   |
                   v
          Evidence selection
                   |
                   v
          Context construction
                   |
                   v
          Language-model generation
                   |
                   v
       Answer + citations + abstention

The system separates retrieval from generation so that retrieval configurations can be evaluated independently before their effect on generated answers is measured.

## 5.2 Document Processing

The corpus consists of synthetic enterprise-style Markdown documents representing operational knowledge such as human resources, security, vendor management, engineering operations, and related enterprise processes.

Documents are segmented using heading-aware chunking. The chunking process preserves document source and section metadata so that retrieval results can be evaluated against section-level ground truth.

The current experimental configuration uses:

- Chunk size: 900 characters
- Chunk overlap: 120 characters
- Metadata: source, section, chunk index, and chunk text

The resulting stress-test corpus contains approximately 100 indexed chunks.

## 5.3 Dense Retrieval

Dense embeddings are generated using OpenAI `text-embedding-3-small` with a configured dimensionality of 512.

The embeddings are stored in a Pinecone vector index using cosine similarity.

For each query, the dense retriever returns a ranked list of candidate passages together with their dense similarity scores.

The primary dense retrieval evaluation examines:

- Top-1 retrieval accuracy
- Recall@3
- Unsupported-query rejection
- Known-query rejection
- Multi-section evidence recall

## 5.4 Sparse Retrieval

A BM25 retriever is constructed over the same corpus.

The sparse retrieval representation includes section names, document identifiers, and chunk text. This allows lexical matches to capture terminology that may be important to enterprise questions.

BM25 results are independently ranked before being combined with dense retrieval results.

## 5.5 Hybrid Retrieval

Dense and sparse results are combined using Reciprocal Rank Fusion.

For a document appearing at rank \(r\), the RRF contribution is:

\[
RRF(d) = \frac{1}{k+r}
\]

where \(k=60\) in the experimental configuration.

For a document appearing in both rankings, the contributions from the two rankings are combined.

The resulting ranking provides a unified candidate list while retaining information about the original dense and sparse rankings.

## 5.6 Evidence Selection

A key experimental variable is the use of dense similarity thresholds.

For threshold-based configurations, a retrieved passage is considered strongly supported by dense retrieval when:

\[
s_{dense} \geq \tau
\]

where \(s_{dense}\) is the dense similarity score and \(\tau\) is the configured threshold.

The hybrid retrieval implementation additionally preserves evidence when a passage is supported by both dense and sparse retrieval rankings. This allows lexical evidence to complement semantic similarity when a passage's dense score alone may not exceed the configured threshold.

The production implementation therefore distinguishes between:

1. Strong dense evidence.
2. Evidence jointly supported by dense and sparse retrieval.

The research evaluation treats these mechanisms as experimental retrieval configurations rather than assuming that the production evidence-selection rule is itself a new retrieval algorithm.

This distinction allows the study to evaluate retrieval behavior while maintaining a clear separation between the experimental benchmark and the production implementation.

## 5.7 Generation

Retrieved contexts are supplied to a language model using a grounded generation prompt.

The generation system is instructed to:

1. Use only the supplied context.
2. Avoid inventing facts or sources.
3. Abstain when the context does not contain sufficient information.
4. Provide citations to supporting source sections.
5. Use all relevant evidence for multi-part questions.
6. Return structured output containing the answer, citations, and abstention status.

The current generation model is `gpt-4o-mini`.

## 5.8 Observability

The production implementation records operational measurements including:

- Total request latency
- Retrieval latency
- Generation latency
- Context count
- Prompt tokens
- Completion tokens
- Total tokens
- Estimated generation cost

These measurements allow retrieval quality to be considered alongside practical system costs.

---

# 6. Dataset and Evaluation Benchmark

## 6.1 Corpus

The benchmark uses 20 synthetic enterprise-style Markdown documents containing approximately 100 retrieval chunks.

Synthetic documents are used to provide explicit ground truth and controlled coverage of enterprise-style concepts. Each document contains structured sections representing specific operational policies or procedures.

The synthetic nature of the corpus is an intentional limitation and is discussed further in Section 12.

## 6.2 Retrieval Benchmark

The retrieval benchmark contains 150 evaluation cases:

- 135 known questions
- 15 unsupported questions
- 15 multi-section questions

Known questions correspond to information present in the corpus.

Unsupported questions are designed so that the corpus does not contain sufficient evidence to answer them.

Multi-section questions require evidence from two or more document sections.

Each evaluation case contains explicit expected evidence targets where applicable.

## 6.3 Answer-Quality Benchmark

A separate 60-case benchmark is used to evaluate the downstream generation system.

It contains:

- 30 known single-section questions
- 15 multi-section questions
- 15 unsupported questions

For known and multi-section questions, the benchmark contains the expected evidence sections used to establish ground-truth context.

## 6.4 Evaluation Separation

Retrieval evaluation and answer evaluation are treated as separate stages.

The retrieval benchmark measures whether the system retrieves the required evidence.

The answer benchmark measures whether the generation system can use the retrieved evidence correctly.

This separation is necessary because a generation failure can originate from either retrieval failure or generation behavior.

---

# 7. Evaluation Metrics

## 7.1 Known Top-1 Accuracy

Known Top-1 measures the percentage of supported questions for which the highest-ranked retrieved result matches an expected evidence target.

\[
Top1 = \frac{\text{correct top-1 cases}}{\text{known cases}}
\]

## 7.2 Recall@3

Recall@3 measures whether the expected evidence is represented within the top three retrieved results.

For single-section cases, a case is considered successfully retrieved when the expected evidence target appears within the top three results.

For multi-section cases, the metric is evaluated against the set of required evidence targets. This allows the evaluation to measure whether the retrieved results provide coverage of the evidence required to answer the question.

## 7.3 Unsupported-Query Rejection

Unsupported-query rejection measures the proportion of unknown questions for which the retrieval system does not provide sufficient evidence to support an answer.

\[
UnknownRejection =
\frac{\text{correctly rejected unknown cases}}
{\text{unknown cases}}
\]

## 7.4 Known-Query Rejection

Known-query rejection measures the proportion of supported questions incorrectly rejected by the evidence-selection mechanism.

This metric captures the cost of aggressive filtering.

## 7.5 Multi-Section Recall

Multi-section recall measures the proportion of required evidence targets recovered for questions requiring information from multiple sections.

This metric is particularly important because a system may achieve strong single-passage retrieval performance while still failing to provide complete evidence for multi-part questions.

## 7.6 Answer Correctness

Answer correctness measures whether the generated answer correctly addresses the question based on the benchmark's expected information.

## 7.7 Faithfulness

Faithfulness measures whether the generated answer is supported by the retrieved evidence rather than introducing unsupported claims.

## 7.8 Citation Accuracy

Citation accuracy measures whether citations included in the generated answer correspond to retrieved evidence that actually supports the associated answer.

## 7.9 Abstention

Abstention measures whether the system appropriately declines to answer when the retrieved context is insufficient.

## 7.10 Latency and Cost

Latency and estimated generation cost are reported to characterize the operational consequences of retrieval configuration.

## 7.11 Statistical Uncertainty

Because the retrieval benchmark contains a finite number of evaluation cases, point estimates are reported together with 95% confidence intervals where appropriate. For binary retrieval metrics, Wilson score intervals are used rather than normal-approximation intervals.

Confidence intervals characterize uncertainty associated with the finite benchmark size and are not interpreted as evidence of statistical significance between retrieval configurations.

The relatively small number of unsupported-query cases also results in wider uncertainty intervals for rejection metrics. These intervals are therefore considered when interpreting the reported results.

For downstream answer-quality metrics, the reported percentages are retained as point estimates because the evaluation uses graded judge scores rather than a simple binary outcome for every metric. Confidence intervals are therefore not reported for these metrics in the current study.
---

# 8. Experimental Design

The experiments compare retrieval configurations incrementally.

## 8.1 Experiment A — Dense Retrieval Baseline

The first experiment evaluates dense semantic retrieval without evidence thresholding.

This establishes the baseline retrieval capability of the semantic retriever.

## 8.2 Experiment B — Dense Retrieval with Similarity Threshold

The second experiment applies similarity thresholds to dense retrieval.

Multiple threshold values are evaluated to characterize the trade-off between supported-query retrieval and unsupported-query rejection.

The threshold sweep evaluates values ranging from 0.20 to 0.65.

## 8.3 Experiment C — Hybrid Retrieval

The third experiment combines dense and BM25 retrieval using reciprocal rank fusion.

This experiment examines whether lexical retrieval provides complementary evidence for enterprise queries.

## 8.4 Experiment D — Hybrid Retrieval with Evidence Selection

The fourth experiment evaluates hybrid retrieval together with evidence selection.

Evidence can be retained either when dense similarity is sufficiently strong or when the passage is supported by both dense and sparse retrieval rankings.

## 8.5 Experiment E — Downstream Answer Evaluation

The final evaluation stage measures the effect of retrieval configuration on generated answers.

The evaluation examines:

- Correctness
- Faithfulness
- Citation accuracy
- Abstention behavior
- Multi-section performance

The purpose is to determine whether retrieval improvements translate into improvements in the final RAG response.

# 9. Results

The experiments evaluate the retrieval system at two levels. The first evaluates retrieval behavior across the 150-case benchmark. The second evaluates the downstream generated responses across the 60-case answer-quality benchmark.

The retrieval benchmark contains 135 supported questions and 15 unsupported questions. The answer-quality benchmark contains 30 supported single-section questions, 15 multi-section questions, and 15 unsupported questions.

## 9.1 Retrieval Results

Table 1 summarizes the retrieval results for the hybrid configurations evaluated in the stress-test benchmark.

### Table 1. Retrieval performance across hybrid configurations

| Configuration | Known Top-1 | Known Recall@3 | Unknown Rejection | Known Rejection | Overall |
|---|---:|---:|---:|---:|---:|
| Hybrid | 100.00% (135/135) | 100.00% (135/135) | 0.00% (0/15) | 0.00% (0/135) | 90.00% |
| Hybrid + Threshold | 98.52% (133/135) | 98.52% (133/135) | 93.33% (14/15) | 1.48% (2/135) | 98.00% |
| Hybrid + Threshold + Reranking | 98.52% (133/135) | 98.52% (133/135) | 93.33% (14/15) | 1.48% (2/135) | 98.00% |

For the hybrid-plus-threshold configuration, the 98.52% known-query Top-1 accuracy (133/135) has a 95% Wilson confidence interval of approximately 94.76%–99.59%. The 93.33% unsupported-query rejection rate (14/15) has a wider 95% interval of approximately 70.18%–98.81%, reflecting the small number of unsupported cases. The overall accuracy of 98.00% (147/150) has a 95% Wilson confidence interval of approximately 94.29%–99.32%.

These intervals illustrate that the unsupported-query rejection estimate has greater uncertainty than the overall benchmark estimate because the unsupported-query subset contains only 15 cases.

The hybrid baseline retrieved the correct evidence for all 135 supported cases in the benchmark. However, without evidence filtering, none of the 15 unsupported questions were rejected. This illustrates an important distinction between retrieval relevance and unsupported-query detection: a retrieval system can consistently identify relevant evidence for supported questions while still returning plausible context for questions outside the knowledge base.

Adding dense similarity-based evidence selection to the hybrid retrieval pipeline changed this behavior substantially. Known-query Top-1 accuracy decreased from 100.00% to 98.52%, corresponding to two of the 135 supported cases being rejected or failing the retrieval criterion. At the same time, unsupported-query rejection increased from 0.00% to 93.33%, with 14 of the 15 unsupported questions correctly rejected.

The resulting overall benchmark accuracy increased from 90.00% for the unfiltered hybrid configuration to 98.00% for the thresholded configuration. This improvement should be interpreted as a benchmark-specific trade-off rather than evidence that a fixed threshold is universally optimal.

The results therefore support the existence of a measurable trade-off between supported-query retrieval coverage and unsupported-query rejection. In this benchmark, the threshold removed a small number of supported cases while substantially reducing unsupported retrieval.

## 9.2 Threshold Trade-off

The threshold experiment provides evidence that retrieval filtering changes the error profile of the RAG system rather than simply increasing or decreasing retrieval quality in one direction.

Without thresholding, the hybrid system retained complete supported-query retrieval performance but did not reject unsupported questions. Relative to the unfiltered hybrid baseline, unsupported-query rejection increased to 93.33%, while supported-query Top-1 accuracy and Recall@3 decreased by 1.48 percentage points.

The two supported cases lost under thresholding are particularly important because they demonstrate the potential cost of using a global similarity criterion for evidence selection. A threshold assumes that the relevance score of an individual passage is sufficient to determine whether that passage should be retained. However, evidence required to answer a question may be distributed across passages whose individual scores do not all exceed the threshold.

Consequently, thresholding should be interpreted as an evidence-control mechanism rather than as a universal indicator of answer sufficiency.

## 9.3 Reranking Result

The evaluated hybrid-plus-threshold-plus-reranking configuration produced the same aggregate retrieval metrics as the hybrid-plus-threshold configuration: 133 of 135 supported cases were correct at Top-1 and Recall@3, while 14 of 15 unsupported cases were rejected.

The reranking experiment was subject to external service rate limitations during experimentation. The available results are therefore treated as exploratory rather than as a controlled comparison of reranking effectiveness. The aggregate benchmark metrics matched the thresholded hybrid configuration, but the experiment does not provide sufficient evidence to determine whether reranking improves retrieval quality under unconstrained conditions.

## 9.4 Downstream Answer Quality

The downstream answer-quality benchmark contains 60 cases: 30 known single-section questions, 15 multi-section questions, and 15 unsupported questions.

### Table 2. Downstream answer-quality results

| Metric | Result |
|---|---:|
| Answer correctness | 94.44% |
| Faithfulness | 93.33% |
| Citation accuracy | 93.33% |
| Unknown-query abstention | 100.00% |
| Multi-section fully correct | 80.00% |
| Multi-section fully faithful | 80.00% |
| Multi-section fully cited | 80.00% |

The answer-quality results indicate that the system generally produced answers that were judged correct and supported by the retrieved context. Correctness was 94.44%, while faithfulness and citation accuracy were both 93.33%.

The unsupported-question subset achieved 100% abstention. This is an important downstream property because the objective of an enterprise RAG system is not simply to maximize answer generation, but to avoid producing unsupported answers when sufficient evidence is unavailable.

## 9.5 Multi-Section Answering

Multi-section questions produced a lower fully successful rate than the overall answer-quality benchmark. Eighty percent of the multi-section cases were fully correct, fully faithful, and fully cited.

The remaining cases illustrate the difficulty of answering questions that depend on multiple pieces of evidence. The evaluation identified three failed multi-section cases involving Kubernetes secrets and workload health, retention and approved deletion requests, and vendor onboarding followed by offboarding.

In these cases, the generated system response abstained, and the available retrieved context was insufficient to support a complete answer according to the evaluation. The failures therefore demonstrate an important distinction between safe abstention and complete evidence retrieval. Abstention can prevent unsupported generation, but it does not compensate for missing evidence when the answer should have been recoverable from the corpus.

The vendor onboarding/offboarding case is particularly relevant to the retrieval research question because it requires evidence from multiple sections of the vendor-management documentation. This type of query illustrates why evaluating only the highest-scoring individual passage can be insufficient for enterprise RAG.

## 9.6 Relationship Between Retrieval and Generation

The retrieval and answer-quality experiments together suggest that retrieval filtering affects downstream RAG behavior through the amount and completeness of evidence supplied to the generator.

The thresholded hybrid configuration improved unsupported-query rejection substantially while accepting a small reduction in supported-query retrieval performance. At the answer level, the system achieved 100% abstention on unsupported questions, while multi-section questions remained a source of failure.

These observations support the broader interpretation that enterprise RAG retrieval should be evaluated using multiple dimensions rather than a single retrieval score. Supported-query accuracy, unsupported-query rejection, evidence coverage, and downstream answer quality capture different aspects of system behavior.

The current benchmark does not establish a causal relationship between any individual retrieval metric and answer quality. Instead, it provides controlled evidence that retrieval configuration and evidence selection affect the amount and completeness of evidence available to the generation model when producing a grounded answer.

# 10. Error Analysis

The error analysis focuses on cases where retrieval or evidence selection produces behavior that cannot be explained by a simple relevance-ranking interpretation.

## 10.1 Semantically Related but Insufficient Evidence

A passage may receive a high semantic similarity score because it discusses a concept related to the question without containing the specific information required to answer it.

These cases illustrate the distinction between semantic relevance and answer-supporting evidence.

## 10.2 Multi-Section Evidence Loss

Some questions require multiple sections to answer completely.

An aggressive similarity threshold can remove one of the required sections even when another relevant section remains available in the broader retrieval candidate set.

This can cause the generation stage to abstain or produce an incomplete response.

## 10.3 Lexical Recovery

Hybrid retrieval can recover passages containing exact terminology that may receive a weaker dense similarity score.

These cases provide evidence for treating semantic and lexical retrieval as complementary signals.

## 10.4 Unsupported Questions

Unsupported questions are particularly important because enterprise RAG systems should avoid generating plausible answers when the knowledge base does not contain sufficient evidence.

The analysis therefore distinguishes:

- correct abstention,
- incorrect evidence retrieval,
- unsupported generation,
- and false rejection of supported questions.

## 10.5 Downstream Generation Failures

Generation failures are analyzed separately from retrieval failures.

If the correct evidence is present but the generated answer is incorrect or inadequately cited, the failure is attributed to downstream generation behavior rather than retrieval.

---

# 11. Discussion

The experimental results indicate that enterprise RAG retrieval should be evaluated as a combination of relevance, evidence sufficiency, and unsupported-query rejection rather than semantic similarity alone.

The unfiltered hybrid configuration achieved 100.00% Top-1 accuracy and 100.00% Recall@3 across the 135 supported retrieval cases. However, because the configuration did not reject unsupported queries, its overall benchmark accuracy was 90.00%.

Introducing threshold-based evidence selection changed this behavior substantially. Unknown-query rejection increased to 93.33%, while supported-query Top-1 accuracy and Recall@3 decreased to 98.52%. The resulting overall benchmark accuracy increased to 98.00%.

These results illustrate the central trade-off investigated in this study. A stricter evidence-selection policy can improve the system's ability to reject unsupported questions, but excessive filtering can also remove evidence that is relevant to supported questions.

The multi-section results provide an additional perspective. Although the hybrid retrieval configuration achieved strong aggregate retrieval performance, fully correct multi-section answers reached 80.00% in the downstream answer-quality benchmark. This indicates that retrieving highly relevant individual passages does not necessarily guarantee that all evidence required for a complete answer will be preserved.

The answer-quality evaluation further supports this observation. Overall answer correctness was 94.44%, while faithfulness and citation accuracy were both 93.33%. Multi-section cases remained more challenging, with 80.00% of cases receiving fully correct, faithful, and cited answers.

These results suggest that RAG evaluation should distinguish between passage relevance and evidence completeness. A passage can be highly relevant to a query without containing all of the information required to answer it.

Hybrid retrieval provides complementary retrieval signals by combining semantic and lexical matching. Dense retrieval can capture conceptual similarity, while BM25 can recover passages containing important exact terminology. However, the experiments do not establish that hybrid retrieval is universally superior across enterprise corpora. The findings are specific to the controlled benchmark evaluated in this study.

An important implication is that future enterprise RAG benchmarks should explicitly measure evidence coverage for multi-part questions in addition to conventional Top-1 and Recall@k metrics.

# 12. Limitations

## 12.1 Synthetic Corpus

The evaluation corpus consists of synthetic enterprise-style documents rather than proprietary production documents.

This provides control over ground truth but limits the ability to generalize results to all enterprise knowledge bases.

## 12.2 Benchmark Size

The retrieval benchmark contains 150 cases and the answer-quality benchmark contains 60 cases.

These sizes are sufficient for controlled experimentation but are not large enough to establish broad statistical generalization across enterprise domains.

## 12.3 Model Dependence

The experiments use a specific embedding model and generation model.

Different embedding models, language models, chunking strategies, or vector databases may produce different results.

## 12.4 Threshold Generalization

Similarity thresholds are dependent on the embedding model, corpus, chunking strategy, and query distribution.

A threshold identified as useful in this benchmark should therefore not be interpreted as a universal threshold for enterprise RAG systems.

## 12.5 LLM-Based Answer Evaluation

The answer-quality evaluation uses an LLM-based judge.

Although structured evaluation criteria are used, automated judging can introduce evaluator bias and may not perfectly reproduce human expert assessment.

## 12.6 Reranking

A neural reranking experiment was considered but excluded from the final comparison because the external reranking service used during experimentation imposed request-rate and usage limitations.

Consequently, the final study focuses on dense, sparse, hybrid, and threshold/evidence-selection configurations.

## 12.7 Domain Distribution

The corpus contains a controlled mixture of enterprise-style topics but does not represent the complete diversity of enterprise documentation.

---

# 13. Future Work

Several extensions are possible.

### 13.1 Larger Enterprise Benchmarks

Future work should evaluate the methodology on larger and more diverse enterprise datasets.

### 13.2 Learned Evidence Selection

Instead of relying on fixed similarity thresholds, future experiments could investigate learned evidence-selection models.

### 13.3 Adaptive Thresholds

An adaptive threshold could account for query type, evidence density, or retrieval-score distributions rather than applying one global threshold.

### 13.4 Query Decomposition

Multi-section questions could be decomposed into subqueries before retrieval.

This may allow the system to independently identify evidence for each required component.

### 13.5 Reranking

Future experiments should evaluate cross-encoder or neural reranking under a controlled and reproducible configuration.

### 13.6 Human Evaluation

Human expert evaluation could complement automated answer judging and provide an independent assessment of correctness and faithfulness.

### 13.7 Statistical Significance

Larger benchmarks would allow confidence intervals and statistical significance testing across retrieval configurations.

### 13.8 Production Enterprise Data

Evaluation on anonymized production enterprise documentation would provide stronger evidence regarding real-world generalization.

---

# 14. Conclusion

This study investigates retrieval and evidence-selection behavior in enterprise Retrieval-Augmented Generation systems.

The central observation motivating the study is that semantic similarity and answer-supporting evidence are related but distinct concepts. A retrieval strategy that optimizes the similarity of individual passages may not necessarily maximize the completeness of the evidence required to answer a question.

By evaluating dense retrieval, similarity thresholds, hybrid dense-sparse retrieval, and downstream answer quality within a controlled benchmark, the study provides a framework for examining this trade-off.

The evaluation explicitly considers supported questions, unsupported questions, and multi-section questions. This enables retrieval behavior to be evaluated not only in terms of whether relevant information appears near the top of a ranking, but also in terms of whether the retrieved context provides sufficient evidence for grounded generation.

The resulting experimental framework provides a reproducible basis for further investigation into adaptive evidence selection, hybrid retrieval, query decomposition, reranking, and enterprise-scale RAG evaluation.

---

# References

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, Tim Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Advances in Neural Information Processing Systems, 2020.

[2] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, "Dense Passage Retrieval for Open-Domain Question Answering," Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 6769–6781, 2020. DOI: 10.18653/v1/2020.emnlp-main.550.

[3] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009. DOI: 10.1561/1500000019.

[4] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759, 2009. DOI: 10.1145/1571941.1572114.

[5] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, "RAGAs: Automated Evaluation of Retrieval Augmented Generation," Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp. 150–158, 2024. DOI: 10.18653/v1/2024.eacl-demo.16.

[6] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 338–354, 2024. DOI: 10.18653/v1/2024.naacl-long.20.

[7] D. Ru, L. Qiu, X. Hu, T. Zhang, P. Shi, S. Chang, C. Jiayang, C. Wang, S. Sun, H. Li, Z. Zhang, B. Wang, J. Jiang, T. He, Z. Wang, P. Liu, Y. Zhang, and Z. Zhang, "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation," Advances in Neural Information Processing Systems 37, 2024. DOI: 10.52202/079017-0692.

[8] J. Chen, H. Lin, X. Han, and L. Sun, "Benchmarking Large Language Models in Retrieval-Augmented Generation," Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, pp. 17754–17762, 2024. DOI: 10.1609/AAAI.V38I16.29728.
