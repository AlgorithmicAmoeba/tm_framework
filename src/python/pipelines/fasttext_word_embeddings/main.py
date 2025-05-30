import fasttext
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm
import configuration as cfg
import database

def embed_all_corpora(session: Session, model):
    """Embed all words in all available corpora"""
    # Get all corpora using SQL
    result = session.execute(text("SELECT DISTINCT corpus_name FROM pipeline.vocabulary_word"))
    corpora = result.fetchall()
    
    for corpus in corpora:
        embed_corpus(session, corpus.corpus_name, model)

def embed_corpus(session: Session, corpus_name: str, model):
    """Embed all words in a specific corpus"""
    # Get vocabulary words using SQL
    result = session.execute(
        text("""
            SELECT id, word 
            FROM pipeline.vocabulary_word 
            WHERE corpus_name = :corpus_name
        """),
        {"corpus_name": corpus_name}
    )
    vocabulary_words = result.fetchall()
    
    # Create embeddings for each word
    pbar = tqdm(total=len(vocabulary_words), desc=f"Embedding words for corpus '{corpus_name}'")
    
    for vocab_word in vocabulary_words:
        try:
            vector = model[vocab_word.word].tolist()  # Convert numpy array to list
            
            # Insert embedding using SQL
            session.execute(
                text("""
                    INSERT INTO pipeline.vocabulary_word_embeddings 
                    (vocabulary_word_id, vector) 
                    VALUES (:vocab_id, :vector)
                    ON CONFLICT (vocabulary_word_id) DO UPDATE 
                    SET vector = :vector
                """),
                {
                    "vocab_id": vocab_word.id,
                    "vector": vector
                }
            )
            
        except KeyError:
            print(f"Warning: Word '{vocab_word.word}' not found in FastText model")
        pbar.update(1)
    
    pbar.close()

def main():
    # Load FastText model
    model = fasttext.load_model("ignore/fasttext/crawl-300d-2M-subword.bin")

    # Load configuration and connect to database
    config = cfg.load_config_from_env()
    db_config = config.database

    with database.get_session(db_config) as session:
        embed_all_corpora(session, model)
        session.commit()

if __name__ == "__main__":
    main()
