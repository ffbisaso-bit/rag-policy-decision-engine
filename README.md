# RAG Policy Decision Engine

A production-style Retrieval-Augmented Generation system designed to evaluate policy evidence before allowing action.

This project shows how controlled language systems move beyond simple retrieval and generation by forcing explicit decision logic before output.

---

## Problem It Solves

Most RAG systems retrieve documents and answer too early.

That creates hidden risk:

- outdated policy answers  
- weak evidence presented confidently  
- contradictions ignored  
- unsafe recommendations delivered fluently  

This system introduces decision control before generation.

---

## Core Architecture

User Query  
→ Query Rewriting  
→ Dense Retrieval  
→ Lexical Reranking  
→ Evidence Assessment  
→ Confidence Gate  
→ Arbitration  
→ Decision Engine  
→ ACT / REFUSE / ESCALATE  
→ Audit Log  

---

## Final Decision States

- ACT → evidence sufficient and safe  
- REFUSE → evidence insufficient  
- ESCALATE → conflict or risk requires higher review  

---

## Main Components

- query_rewriter.py  
- vector_store.py  
- evidence_assessor.py  
- confidence_gate.py  
- arbitrator.py  
- decision_engine.py  
- answer_generator.py  
- audit_log.py  

---

## Why This Matters

In production AI, retrieval alone is not enough.

The critical problem is deciding when evidence is strong enough to justify action.

This project models that decision explicitly.

---

## Status

Production-ready controlled AI decision system. Acts when evidence supports it. Refuses when it doesn’t. Escalates when certainty is insufficient.
