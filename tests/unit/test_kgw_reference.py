import math
import random

from provenance.detectors.reference.kgw import KGWReferenceConfig, KGWReferenceScorer


def _reference_permutation(seed, vocab_size):
    values = list(range(vocab_size))
    random.Random(int(seed)).shuffle(values)
    return values


def _reference_prf(config, vocab_size):
    return _reference_permutation(config.hash_key, vocab_size)


def _reference_f(config, vocab_size, context):
    prefix = context[-config.prefix_length :]
    prf = _reference_prf(config, vocab_size)
    if config.f_scheme == "additive":
        return prf[sum(prefix) % vocab_size]
    if config.f_scheme == "time":
        value = 1
        for token_id in prefix:
            value *= token_id
        return prf[value % vocab_size]
    if config.f_scheme == "skip":
        return prf[prefix[0] % vocab_size]
    return min(prf[token_id % vocab_size] for token_id in prefix)


def _reference_greenlist(config, vocab_size, context):
    seed = (config.hash_key * _reference_f(config, vocab_size, context)) % vocab_size
    size = int(vocab_size * config.gamma)
    return _reference_permutation(seed, vocab_size)[:size]


def _reference_score(config, vocab_size, token_ids):
    green = 0
    scored = 0
    flags = []
    for index in range(config.prefix_length, len(token_ids)):
        context = token_ids[index - config.prefix_length : index]
        is_green = token_ids[index] in _reference_greenlist(config, vocab_size, context)
        flags.append(is_green)
        scored += 1
        green += int(is_green)
    z_score = (green - config.gamma * scored) / math.sqrt(scored * config.gamma * (1 - config.gamma))
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2))
    return green, scored, flags, z_score, p_value


def _config():
    return KGWReferenceConfig(
        configuration_id="unit-kgw",
        version="1",
        gamma=0.25,
        delta=2.0,
        hash_key=15485863,
        z_threshold=4.0,
        prefix_length=2,
        f_scheme="additive",
        window_scheme="left",
        min_scored_tokens=2,
    )


def test_kgw_reference_greenlist_matches_independent_reference():
    config = _config()
    scorer = KGWReferenceScorer(config, vocab_size=16)

    assert scorer.greenlist_ids([3, 7]) == _reference_greenlist(config, 16, [3, 7])


def test_kgw_reference_score_matches_independent_reference():
    config = _config()
    scorer = KGWReferenceScorer(config, vocab_size=16)
    token_ids = [1, 4, 7, 3, 2, 15, 8, 8, 11, 0, 5, 9]

    expected_green, expected_scored, expected_flags, expected_z, expected_p = _reference_score(
        config,
        16,
        token_ids,
    )
    score = scorer.score_token_ids(token_ids)

    assert score.green_token_count == expected_green
    assert score.scored_token_count == expected_scored
    assert score.green_token_mask == expected_flags
    assert score.z_score == expected_z
    assert score.p_value == expected_p


def test_kgw_reference_can_ignore_repeated_ngrams():
    config = KGWReferenceConfig(
        configuration_id="unit-kgw",
        version="1",
        gamma=0.25,
        delta=2.0,
        hash_key=15485863,
        z_threshold=4.0,
        prefix_length=1,
        f_scheme="time",
        window_scheme="left",
        min_scored_tokens=1,
        ignore_repeated_ngrams=True,
    )
    scorer = KGWReferenceScorer(config, vocab_size=8)

    score = scorer.score_token_ids([1, 2, 1, 2, 1, 2])

    assert score.scored_token_count == 2
    assert score.ignored_repeated_count == 3
    assert score.green_token_mask.count(None) == 3
