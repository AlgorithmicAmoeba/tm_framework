from sqlalchemy import text
import hashlib

def generate_content_hash(content: str) -> str:
    """Generates an SHA256 hash for the given content."""
    if content is None:
        return None
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def process_and_fill_vocabulary_documents_textual(session):
    """
    Finds pre-processed documents corpus by corpus, filters their content using
    corpus-specific vocabulary words, and fills the vocabulary_document table
    if the raw_document_hash doesn't already exist for that corpus,
    using textual SQL queries. Commits are done per corpus.

    Args:
        session: A SQLAlchemy session object.
    """
    total_new_vocab_docs_across_all_corpora = 0

    try:
        print("Fetching distinct corpus names from pre-processed documents...")
        distinct_corpus_names_result = session.execute(
            text(f"SELECT DISTINCT corpus_name FROM pipeline.preprocessed_document")
        )
        distinct_corpus_names = [row.corpus_name for row in distinct_corpus_names_result.fetchall()]

        if not distinct_corpus_names:
            print("No corpus names found in pre-processed documents. Exiting.")
            return

        print(f"Found {len(distinct_corpus_names)} distinct corpus names: {distinct_corpus_names}")

        insert_statement_sql = f"""
            INSERT INTO pipeline.vocabulary_document
                (raw_document_hash, corpus_name, content, content_hash, created_at, updated_at)
            VALUES
                (:raw_document_hash, :corpus_name, :content, :content_hash, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        insert_statement = text(insert_statement_sql)

        for current_corpus_name in distinct_corpus_names:
            print(f"\n--- Processing corpus: {current_corpus_name} ---")
            new_docs_for_this_corpus = 0

            try:
                # 1. Fetch vocabulary for the current corpus
                print(f"Fetching vocabulary for corpus '{current_corpus_name}'...")
                vocab_words_result = session.execute(
                    text(f"SELECT word FROM pipeline.vocabulary_word WHERE corpus_name = :corpus_name"),
                    {'corpus_name': current_corpus_name}
                )
                current_corpus_vocabulary = {row.word.lower() for row in vocab_words_result.fetchall()}

                if not current_corpus_vocabulary:
                    print(f"No vocabulary words found for corpus '{current_corpus_name}'. Skipping this corpus.")
                    continue
                print(f"Vocabulary loaded for corpus '{current_corpus_name}': {len(current_corpus_vocabulary)} words.")

                # 2. Fetch existing vocabulary document hashes for the current corpus
                print(f"Fetching existing vocabulary document hashes for corpus '{current_corpus_name}'...")
                existing_vocab_keys_result = session.execute(
                    text(f"SELECT raw_document_hash FROM pipeline.vocabulary_document WHERE corpus_name = :corpus_name"),
                    {'corpus_name': current_corpus_name}
                )
                existing_vocab_doc_hashes_for_corpus = {row.raw_document_hash for row in existing_vocab_keys_result.fetchall()}
                print(f"Found {len(existing_vocab_doc_hashes_for_corpus)} existing vocabulary document entries for corpus '{current_corpus_name}'.")

                # 3. Fetch pre-processed documents for the current corpus
                print(f"Fetching pre-processed documents for corpus '{current_corpus_name}'...")
                preprocessed_docs_result = session.execute(
                    text(f"SELECT raw_document_hash, content FROM pipeline.preprocessed_document WHERE corpus_name = :corpus_name"),
                    {'corpus_name': current_corpus_name}
                )
                preprocessed_docs_for_corpus = preprocessed_docs_result.fetchall()

                if not preprocessed_docs_for_corpus:
                    print(f"No pre-processed documents found for corpus '{current_corpus_name}'.")
                    continue
                print(f"Found {len(preprocessed_docs_for_corpus)} pre-processed documents for corpus '{current_corpus_name}'.")

                # 4. Process documents for the current corpus
                for pp_doc_row in preprocessed_docs_for_corpus:
                    if pp_doc_row.raw_document_hash in existing_vocab_doc_hashes_for_corpus:
                        # print(f"Skipping document: Hash '{pp_doc_row.raw_document_hash}' in corpus '{current_corpus_name}' already exists in vocabulary_document.")
                        continue

                    doc_content = pp_doc_row.content
                    filtered_content = "" # Default to empty string

                    if not doc_content:
                        print(f"Warning: Pre-processed document with hash '{pp_doc_row.raw_document_hash}' in corpus '{current_corpus_name}' has no content.")
                        # filtered_content remains ""
                    else:
                        words_in_doc = doc_content.lower().split() # Simple whitespace tokenization
                        vocabulary_content_words = [word for word in words_in_doc if word in current_corpus_vocabulary]
                        filtered_content = " ".join(vocabulary_content_words)

                    new_content_hash = generate_content_hash(filtered_content)

                    params = {
                        'raw_document_hash': pp_doc_row.raw_document_hash,
                        'corpus_name': current_corpus_name,
                        'content': filtered_content,
                        'content_hash': new_content_hash
                    }
                    session.execute(insert_statement, params)
                    new_docs_for_this_corpus += 1
                    # print(f"Prepared insert for new vocabulary document: Corpus '{current_corpus_name}', Hash '{pp_doc_row.raw_document_hash}'")

                if new_docs_for_this_corpus > 0:
                    session.commit()
                    print(f"Successfully committed {new_docs_for_this_corpus} new documents for corpus '{current_corpus_name}'.")
                    total_new_vocab_docs_across_all_corpora += new_docs_for_this_corpus
                else:
                    print(f"No new documents were added for corpus '{current_corpus_name}'.")

            except Exception as e_corpus:
                session.rollback()
                print(f"An error occurred while processing corpus '{current_corpus_name}': {e_corpus}")
                print(f"Rolled back changes for corpus '{current_corpus_name}'. Continuing with the next corpus if any.")
                # Optionally, re-raise if you want the whole script to stop on any corpus error:
                # raise

        print(f"\n--- Processing Summary ---")
        if total_new_vocab_docs_across_all_corpora > 0:
            print(f"Successfully added a total of {total_new_vocab_docs_across_all_corpora} new documents to pipeline.vocabulary_document across all processed corpora.")
        else:
            print(f"No new documents were added to pipeline.vocabulary_document in this run.")

    except Exception as e_main:
        # This outer rollback would only be effective if an error occurs outside
        # the per-corpus try-except (e.g., fetching distinct corpus names)
        # or if a per-corpus error is re-raised.
        session.rollback()
        print(f"A critical error occurred during the overall process: {e_main}")
        raise
    # Session closing is handled by the 'with' statement in the __main__ block

if __name__ == '__main__':
    import configuration
    import database

    print(f"Using schema: pipeline")

    try:
        config = configuration.load_config_from_env()
        db_config = config.database # Make sure this path is correct for your config object
        
        with database.get_session(db_config) as session:
            process_and_fill_vocabulary_documents_textual(session)
            
    except FileNotFoundError as e_conf:
        print(f"Configuration file error: {e_conf}. Ensure 'configuration.py' and necessary env vars are set.")
    except AttributeError as e_attr:
        print(f"Configuration object error: {e_attr}. Check the structure of your 'config.database' object.")
    except ImportError as e_imp:
        print(f"Import error: {e_imp}. Ensure 'configuration.py' and 'database.py' are in your PYTHONPATH or current directory.")
    except Exception as e:
        print(f"An unexpected error occurred in __main__: {e}")