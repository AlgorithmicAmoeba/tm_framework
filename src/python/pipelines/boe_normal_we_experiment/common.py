"""
Shared configuration for the BOE "normal word-embedding" ablation experiment.

This experiment reuses the BOE@1 document representation (first-chunk
all-MiniLM-L6-v2 embeddings already stored in
pipeline.boe_cse_document_embedding at target_chunk_count=1) but replaces the
BOE-derived (TF-IDF + Ridge) word embeddings with "normal" word embeddings:
each vocabulary word encoded DIRECTLY with SentenceTransformer('all-MiniLM-L6-v2')
into the SAME 384-d space as the document embeddings.

Only the two word-embedding-driven topic-model families are trained: KeyNMF and
SemanticSignalSeparation (S^3). All experiment-owned tables are prefixed
boe_nwe_.
"""

# --- The encoder used both for the reused doc embeddings and the normal word
#     embeddings. Encoding words DIRECTLY with this model puts them in the same
#     space as the doc embeddings (the whole point of the ablation).
SOURCE_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Doc-embedding slice to REUSE from pipeline.boe_cse_document_embedding ----
DOC_ALGORITHM = "none"
DOC_TARGET_DIMS = 0
DOC_PADDING_METHOD = "noise_only"
DOC_TARGET_CHUNK_COUNT = 1

# --- Topic-model sweep -------------------------------------------------------
FAMILIES = ["KeyNMF", "SemanticSignalSeparation"]
NUM_TOPICS_LIST = [10, 20, 50, 100, 200]
TARGET_RESULTS = 5

# --- Table names -------------------------------------------------------------
DOC_EMBEDDING_TABLE = "pipeline.boe_cse_document_embedding"   # reused input
WORD_EMBEDDING_TABLE = "pipeline.boe_nwe_word_embedding"
TOPIC_MODEL_RESULT_TABLE = "pipeline.boe_nwe_topic_model_corpus_result"
PERFORMANCE_TABLE = "pipeline.boe_nwe_topic_model_performance"
