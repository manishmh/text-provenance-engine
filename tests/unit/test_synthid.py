from provenance.detectors.synthid import SynthIDTextDetector, generate_controlled_synthid_text


CONFIG_PATH = "configs/synthid.example.json"


def test_synthid_controlled_watermarked_positive():
    detector = SynthIDTextDetector.from_config_file(CONFIG_PATH)
    text = generate_controlled_synthid_text(detector, token_count=180, watermarked=True)
    result = detector.detect(text)
    assert result.status == "ok"
    assert result.implementation_kind == "simulation"
    assert result.compatibility == "controlled-local-only"
    assert result.detected is True
    assert result.score >= result.threshold
    assert result.evidence["usable_token_count"] >= 100
    assert result.evidence["scorer"] == "weighted_mean"


def test_synthid_controlled_unwatermarked_negative():
    detector = SynthIDTextDetector.from_config_file(CONFIG_PATH)
    text = generate_controlled_synthid_text(detector, token_count=180, watermarked=False)
    result = detector.detect(text)
    assert result.status == "ok"
    assert result.implementation_kind == "simulation"
    assert result.compatibility == "controlled-local-only"
    assert result.detected is False
    assert result.score < result.threshold
    assert result.evidence["usable_token_count"] >= 100


def test_synthid_short_text_is_insufficient():
    detector = SynthIDTextDetector.from_config_file(CONFIG_PATH)
    text = generate_controlled_synthid_text(detector, token_count=8, watermarked=True)
    result = detector.detect(text)
    assert result.status == "insufficient_text"
    assert result.detected is None


def test_synthid_different_lengths():
    detector = SynthIDTextDetector.from_config_file(CONFIG_PATH)
    for token_count in [50, 240]:
        watermarked = generate_controlled_synthid_text(
            detector,
            token_count=token_count,
            watermarked=True,
        )
        unwatermarked = generate_controlled_synthid_text(
            detector,
            token_count=token_count,
            watermarked=False,
        )
        assert detector.detect(watermarked).detected is True
        assert detector.detect(unwatermarked).detected is False
