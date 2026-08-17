import json
import subprocess
import sys

from provenance import ProvenanceEngine
from provenance.detectors.kgw import KGWDetector, generate_controlled_kgw_text


def test_engine_returns_structured_detector_results():
    engine = ProvenanceEngine()
    result = engine.analyze("hello\u200b world")
    assert result.status == "ok"
    assert result.text_stats["token_count"] == 2
    assert [item.detector for item in result.results] == ["unicode"]
    assert result.results[0].detected is True
    assert "AI-generated" in result.limitations[1]


def test_engine_accepts_watermark_detectors():
    detector = KGWDetector.from_config_file("configs/kgw.example.json")
    text = generate_controlled_kgw_text(detector, token_count=120, watermarked=True)
    result = ProvenanceEngine().analyze(text, detectors=[detector])
    by_name = {item.detector: item for item in result.results}
    assert by_name["unicode"].detected is False
    assert by_name["kgw"].detected is True


def test_cli_json_unicode_fixture():
    completed = subprocess.run(
        [sys.executable, "-m", "provenance", "analyze", "fixtures/unicode.txt", "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    unicode_result = payload["results"][0]
    assert unicode_result["detector"] == "unicode"
    assert unicode_result["detected"] is True
    categories = {finding["category"] for finding in unicode_result["evidence"]["findings"]}
    assert {"zero_width", "bidi_control", "unusual_whitespace"} <= categories


def test_cli_kgw_json(tmp_path):
    detector = KGWDetector.from_config_file("configs/kgw.example.json")
    text = generate_controlled_kgw_text(detector, token_count=140, watermarked=True)
    sample = tmp_path / "kgw.txt"
    sample.write_text(text, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "provenance",
            "analyze",
            str(sample),
            "--detector",
            "kgw",
            "--config",
            "configs/kgw.example.json",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["results"][0]["detector"] == "kgw"
    assert payload["results"][0]["implementation_kind"] == "simulation"
    assert payload["results"][0]["compatibility"] == "controlled-local-only"
    assert payload["results"][0]["detected"] is True
