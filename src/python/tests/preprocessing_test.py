import concurrent.futures
from preprocessing import UnicodeTokenizer

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
