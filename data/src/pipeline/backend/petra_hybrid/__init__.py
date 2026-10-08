"""PETRA-RAG — Hybrid-Retrieval-Bausteine.

Implementiert die Teile der Architektur, für die es keine n8n- oder
LangChain-Nodes gibt (BM25, RRF, CRAG-Schwellwertlogik,
Tabellenserialisierung). Aufruf über FastAPI-Endpoints
(backend/query_router_v3.py) aus n8n-"HTTP Request"-Nodes; Klassifikation,
Prompting und LLM-Aufruf laufen in n8n.
"""