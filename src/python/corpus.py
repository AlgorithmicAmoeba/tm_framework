from sqlalchemy.orm import Session
from models import Corpus as CorpusModel
from models import Document, DocumentType, Embedding, Embedder, VocabularyWord

class Corpus:
    def __init__(self, session: Session, name: str):
        """Initialize a corpus by its name."""
        self.session = session
        self.corpus = self.session.query(CorpusModel).filter_by(name=name).first()
        if self.corpus is None:
            raise ValueError(f"Corpus '{name}' not found")
    
    def get_raw_documents(self) -> list[Document]:
        """Get all raw documents in the corpus."""
        return (
            self.session.query(Document)
            .join(Document.document_type)
            .filter(Document.corpus_id == self.corpus.id)
            .filter(DocumentType.name == 'raw')
            .all()
        )
    
    def get_preprocessed_documents(self) -> list[Document]:
        """Get all preprocessed documents in the corpus."""
        return (
            self.session.query(Document)
            .join(Document.document_type)
            .filter(Document.corpus_id == self.corpus.id)
            .filter(DocumentType.name == 'preprocessed')
            .all()
        )
    
    def get_vocabulary_documents(self) -> list[Document]:
        """Get all vocabulary-only documents in the corpus."""
        return (
            self.session.query(Document)
            .join(Document.document_type)
            .filter(Document.corpus_id == self.corpus.id)
            .filter(DocumentType.name == 'vocabulary_only')
            .all()
        )

    def get_document_vectors(self, embedder_name: str) -> list[Embedding]:
        """Get document vectors for a specific embedder."""
        return (
            self.session.query(Embedding)
            .join(Embedding.document)
            .join(Embedding.embedder)
            .join(Document.document_type)
            .filter(Document.corpus_id == self.corpus.id)
            .filter(Embedder.name == embedder_name)
            .filter(DocumentType.name == 'raw')
            .all()
        )

    def get_vocabulary(self) -> list[str]:
        """Get all vocabulary words in the corpus, ordered by word_index."""
        return (
            self.session.query(VocabularyWord)
            .filter_by(corpus_id=self.corpus.id)
            .order_by(VocabularyWord.word_index)
            .all()
        )
