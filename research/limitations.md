# Limitations

## Synthetic Enterprise Corpus

The evaluation corpus is synthetic and designed to resemble enterprise knowledge rather than being collected from a real organization's internal documentation.

Therefore, the results should not be interpreted as evidence that the observed configuration is universally optimal for all enterprise RAG workloads.

## Dataset Size

The retrieval benchmark contains 150 cases and the answer-quality benchmark contains 60 cases.

Larger and more diverse datasets would provide stronger statistical evidence.

## LLM-Based Answer Evaluation

Answer correctness, faithfulness, and citation accuracy are evaluated using an LLM-based judge.

Although this provides scalable evaluation, model-based judging can introduce evaluator bias and should ideally be complemented by human evaluation.

## Embedding Model

The study uses OpenAI `text-embedding-3-small` with 512 dimensions.

Results may differ with other embedding models or embedding dimensions.

## Generation Model

The answer-generation layer uses a single configured OpenAI chat model.

Results may vary across different language models.

## Reranking

A Pinecone reranking model was investigated but was not included in the final production evaluation because of service rate limitations encountered during experimentation.

Consequently, conclusions about reranking are outside the scope of this study.

## Domain Distribution

The benchmark covers enterprise-style operational topics but does not represent every enterprise domain.

Additional experiments involving legal, financial, healthcare, software engineering, and regulatory documentation would strengthen generalizability.

## Threshold Generalization

Similarity thresholds are dependent on the embedding model, corpus, query distribution, and similarity metric.

The threshold values observed in this study should therefore be treated as experimental configurations rather than universal recommendations.

## Future Work

Future research can evaluate:

- larger real-world enterprise corpora
- additional embedding models
- cross-encoder reranking
- long-context retrieval
- query decomposition
- adaptive retrieval thresholds
- human evaluation
- multilingual enterprise RAG
- agentic retrieval strategies
- temporal and version-aware retrieval