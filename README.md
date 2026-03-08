# RAG Policy Decision Engine

A production-style Retrieval-Augmented Generation system built to evaluate policy evidence before producing decisions.

This project demonstrates how controlled language systems move beyond simple retrieval and generation by enforcing explicit decision layers before action.

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

## Main Components

- query_rewriter.py  
- vector_store.py  
- evidence_assessor.py  
- confidence_gate.py  
- arbitrator.py  
- decision_engine.py  
- answer_generator.py  
- audit_log.py  