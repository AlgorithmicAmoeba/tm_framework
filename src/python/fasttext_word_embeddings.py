import fasttext
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import insert
from tqdm import tqdm

from models import Corpus, VocabularyWord, Embedder, VocabularyWordEmbedding

class FastTextEmbedder:
    def __init__(self, model):
        self.model = model

    def embed_all_corpora(self, session: Session):
        """Embed all words in all available corpora"""
        corpora = session.query(Corpus).all()
        for corpus in corpora:
            self.embed_corpus(session, corpus.name)

    def embed_corpus(self, session: Session, corpus_name: str):
        """Embed all words in a specific corpus"""
        # Get corpus and its vocabulary words
        corpus = session.query(Corpus).filter_by(name=corpus_name).first()
        if not corpus:
            raise ValueError(f"Corpus '{corpus_name}' not found")

        # Get or create FastText embedder entry
        embedder = session.query(Embedder).filter_by(name='fasttext').first()

        # Get vocabulary words
        vocabulary_words = session.query(VocabularyWord).filter_by(corpus_id=corpus.id).all()
        
        # Create embeddings for each word
        pbar = tqdm(total=len(vocabulary_words), desc=f"Embedding words for corpus '{corpus_name}'")
        embeddings = []
        
        for vocab_word in vocabulary_words:
            vector = self.model[vocab_word.word].tolist()  # Convert numpy array to list
            embeddings.append(dict(
                vocabulary_word_id=vocab_word.id,
                embedder_id=embedder.id,
                vector=vector
            ))
            pbar.update(1)
        
        # Store embeddings in database
        self.save_embeddings(session, embeddings)
        pbar.close()

    def save_embeddings(self, session: Session, embeddings: list):
        """Save embeddings to database"""
        session.execute(insert(VocabularyWordEmbedding), embeddings)


def main():
    import configuration as cfg
    import database

    model = fasttext.load_model("ignore/fasttext/crawl-300d-2M-subword.bin")

    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        embedder = FastTextEmbedder(model)
        embedder.embed_all_corpora(session)

        session.commit()


if __name__ == "__main__":
    main()
