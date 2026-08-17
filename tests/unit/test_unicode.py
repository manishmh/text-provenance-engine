from provenance.detectors.unicode import UnicodeArtifactDetector
from provenance.preprocessing.unicode import analyze_unicode


def test_unicode_normal_text_has_no_findings():
    findings = analyze_unicode("This is normal text with ASCII punctuation.")
    assert findings == []


def test_unicode_zero_width_position_and_metadata():
    findings = analyze_unicode("ab\u200bcd")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.position == 2
    assert finding.codepoint == "U+200B"
    assert finding.name == "ZERO WIDTH SPACE"
    assert finding.category == "zero_width"
    assert finding.severity == "medium"


def test_unicode_bidi_position_and_metadata():
    findings = analyze_unicode("a\u202eb")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.position == 1
    assert finding.codepoint == "U+202E"
    assert finding.name == "RIGHT-TO-LEFT OVERRIDE"
    assert finding.category == "bidi_control"
    assert finding.severity == "high"


def test_unicode_unusual_whitespace_position_and_metadata():
    findings = analyze_unicode("a\u00a0b")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.position == 1
    assert finding.codepoint == "U+00A0"
    assert finding.name == "NO-BREAK SPACE"
    assert finding.category == "unusual_whitespace"


def test_unicode_detector_reports_structured_findings():
    result = UnicodeArtifactDetector().detect("x\u200by\u202ez")
    assert result.detector == "unicode"
    assert result.detected is True
    assert result.status == "ok"
    assert result.score == 2
    assert result.evidence["by_category"] == {"zero_width": 1, "bidi_control": 1}
    assert result.evidence["findings"][0]["position"] == 1
