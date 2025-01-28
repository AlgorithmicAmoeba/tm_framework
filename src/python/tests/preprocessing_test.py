import concurrent.futures
from preprocessing import UnicodeTokenizer, Vocabulariser

def test_tokenizer():
    tokenizer = UnicodeTokenizer()

    # Test cases
    test_cases = [
        ("Here's an example: text with numbers 123 and punctuation!", ["Heres", "an", "example", "text", "with", "numbers", "and", "punctuation"]),
        ("Another text, with more punctuation... and numbers 4567.", ["Another", "text", "with", "more", "punctuation", "and", "numbers"]),
        ("No numbers or punctuation here", ["No", "numbers", "or", "punctuation", "here"]),
        ("1234", []),
        ("It's a test!", ["Its", "a", "test"]),
        ("Whitespace    test", ["Whitespace", "test"]),
        ("", []),
    ]

    for text, expected_tokens in test_cases:
        tokenized_text = tokenizer.tokenize(text)
        assert tokenized_text == expected_tokens, f"Tokenization failed for text: {text}"

def test_parallel_processing():
    tokenizer = UnicodeTokenizer()
    texts = [
        "Here's an example: text with numbers 123 and punctuation!",
        "Another text, with more punctuation... and numbers 4567."
    ]
    expected_results = [
        ["Heres", "an", "example", "text", "with", "numbers", "and", "punctuation"],
        ["Another", "text", "with", "more", "punctuation", "and", "numbers"]
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = tokenizer.process_texts(texts, executor)

    assert results == expected_results


def test_vocabulariser():
    tokenizer = UnicodeTokenizer()
    vocabulariser = Vocabulariser(top_n=3)

    # Tokenize the texts
    texts = [
        "Here's an example: text with numbers 123 and punctuation!",
        "Another text, with more punctuation... and numbers 4567."
    ]

    expected_vocabulary = {"heres", "example", "more"}
    expected_transformed_texts = [
        ["heres", "example"],
        ["more"]
    ]
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]

    # Fit the vocabulariser
    vocabulariser.fit_transform(tokenized_texts)

    # Transform the texts
    transformed_texts = vocabulariser.transform(tokenized_texts)

    # Check that the vocabulary contains the top N words
    assert vocabulariser.vocabulary == expected_vocabulary

    # Check that the transformed texts contain only the top N words
    assert transformed_texts == expected_transformed_texts

    # Check that the transformed texts contain only the top N words
    for tokens in transformed_texts:
        for token in tokens:
            assert token in vocabulariser.vocabulary
