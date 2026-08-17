from provenance.detectors.kgw import KGWDetector, generate_controlled_kgw_text


CONFIG_PATH = "configs/kgw.example.json"


def test_kgw_controlled_watermarked_positive():
    detector = KGWDetector.from_config_file(CONFIG_PATH)
    text = generate_controlled_kgw_text(detector, token_count=160, watermarked=True)
    result = detector.detect(text)
    assert result.status == "ok"
    assert result.implementation_kind == "simulation"
    assert result.compatibility == "controlled-local-only"
    assert result.detected is True
    assert result.score >= result.threshold
    assert result.evidence["green_token_count"] > result.evidence["scored_token_count"] * 0.9


def test_kgw_controlled_unwatermarked_negative():
    detector = KGWDetector.from_config_file(CONFIG_PATH)
    text = generate_controlled_kgw_text(detector, token_count=160, watermarked=False)
    result = detector.detect(text)
    assert result.status == "ok"
    assert result.implementation_kind == "simulation"
    assert result.compatibility == "controlled-local-only"
    assert result.detected is False
    assert result.score < result.threshold
    assert result.evidence["green_token_count"] == 0


def test_kgw_short_text_is_insufficient():
    detector = KGWDetector.from_config_file(CONFIG_PATH)
    text = generate_controlled_kgw_text(detector, token_count=8, watermarked=True)
    result = detector.detect(text)
    assert result.status == "insufficient_text"
    assert result.detected is None
    assert result.confidence == "insufficient_evidence"


def test_kgw_long_and_truncated_watermarked_samples_remain_positive():
    detector = KGWDetector.from_config_file(CONFIG_PATH)
    long_text = generate_controlled_kgw_text(detector, token_count=500, watermarked=True)
    long_result = detector.detect(long_text)
    assert long_result.detected is True

    truncated_text = " ".join(long_text.split()[:60])
    truncated_result = detector.detect(truncated_text)
    assert truncated_result.status == "ok"
    assert truncated_result.detected is True
