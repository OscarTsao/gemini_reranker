from criteriabind.text_utils import chunk_text, normalize_whitespace, sentence_tokenize


def test_sentence_tokenize_basic() -> None:
    text = "Sentence one. Sentence two!"
    sentences = sentence_tokenize(text)
    assert sentences == ["Sentence one.", "Sentence two!"]


def test_chunk_text() -> None:
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_text(text, max_tokens=5, stride=5)
    assert len(chunks) >= 4


def test_normalize_whitespace() -> None:
    text = "Hello   world"
    assert normalize_whitespace(text) == "Hello world"
