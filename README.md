# Local RAG Desk

A production-oriented AI research and engineering knowledge agent for a curated corpus of AI, LLM, RAG, and Agentic AI technical documents.

## Objective

The system is designed to answer increasingly complex technical questions using:

- Advanced retrieval
- Agentic research workflows
- Self-hosted LLM inference
- Evidence verification
- Citations
- Memory
- Evaluation
- Observability

The system will be developed incrementally, starting with a baseline RAG implementation and introducing advanced techniques only when they provide measurable value.

## Target Corpus

The initial corpus will contain approximately 100–300 high-quality technical documents covering:

- RAG and retrieval
- Embeddings and vector search
- HNSW and indexing
- LLM serving and inference
- KV caching and batching
- Quantization
- Agentic AI
- Memory
- RAG and LLM evaluation
- Hallucination and faithfulness

## Architecture Evolution

```text
V0 — Baseline RAG
        ↓
V1 — HNSW + Retrieval Evaluation
        ↓
V2 — Hybrid Retrieval + Reranking + HyDE
        ↓
V3 — Semantic Sharding + Advanced Retrieval
        ↓
V4 — LangGraph Agentic Research
        ↓
V5 — vLLM + KV Cache + Batching
        ↓
V6 — Memory + Observability
        ↓
V7 — Production Hardening + CI/CD