from provenance.tokenizers import SimpleVocabularyTokenizer


def test_simple_vocabulary_tokenizer_preserves_ids_and_identity():
    tokenizer = SimpleVocabularyTokenizer(
        vocabulary=("alpha", "beta", "gamma"),
        tokenizer_identifier="unit-simple",
        model_identifier="unit-model",
    )

    token_ids = tokenizer.encode("alpha gamma beta")

    assert token_ids == [0, 2, 1]
    assert tokenizer.decode(token_ids) == "alpha gamma beta"
    assert tokenizer.vocabulary_size == 3
    assert tokenizer.identifier == "unit-simple"
    assert tokenizer.model_identifier == "unit-model"
