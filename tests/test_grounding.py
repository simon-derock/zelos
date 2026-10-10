"""Grounding gate tests (no network)."""
import json
from pathlib import Path

import pytest

from zelos.agent import calc
from zelos.grounding import gate, ledger, numeric
from zelos.grounding.extract import extract_claims, parse_response

FX = Path(__file__).parent / "fixtures"
QC = json.loads((FX / "qc_example.json").read_text())
EV = json.loads((FX / "evidence_example.json").read_text())
TID = QC["tool_call_id"]


@pytest.fixture(autouse=True)
def _no_vectors(monkeypatch):
    monkeypatch.setenv("ZELOS_GATE_VECTORS", "0")


def _verify(claims):
    return gate.verify(claims, [QC], EV)["claims"]


def test_correct_numeric_claim_passes():
    out = _verify([{"text": f"The focus metric is 0.41 with z = 4.2 [{TID}].", "type": "numeric", "evidence": [TID]}])
    assert out[0]["status"] == "supported"


def test_fabricated_number_blocked():
    out = _verify([{"text": f"The focus metric is 0.73 [{TID}].", "type": "numeric", "evidence": [TID]}])
    assert out[0]["status"] == "unsupported" and "0.73" in out[0]["reason"]


def test_percent_vs_fraction():
    leaves = [("m", 0.63)]
    n = numeric.extract_numbers("confluence is 63%")[0]
    assert numeric.match_number(n, leaves) == "m"
    n2 = numeric.extract_numbers("confluence is 0.63")[0]
    assert numeric.match_number(n2, leaves) == "m"
    assert numeric.match_number(numeric.extract_numbers("confluence is 64%")[0], leaves) is None


def test_rounding_and_ids_ignored():
    assert numeric.extract_numbers("Image 220429_01 [qc-1a2b3c4d] PMC123::s1::0 3D") == []
    assert numeric.match_number(numeric.extract_numbers("4.24")[0], [("z", 4.2371)]) == "z"
    assert numeric.match_number(numeric.extract_numbers("4.3")[0], [("z", 4.2371)]) is None


def test_fake_citation_blocked():
    out = _verify([{"text": "Blur lowers the Laplacian variance [PMC999999::s1::0].", "type": "literature",
                    "evidence": ["PMC999999::s1::0"]}])
    assert out[0]["status"] == "unsupported" and "PMC999999" in out[0]["reason"]


def test_uncited_literature_blocked():
    out = _verify([{"text": "Autofocus keeps chips in the focal plane.", "type": "literature", "evidence": []}])
    assert out[0]["status"] == "unsupported"


def test_real_citation_wrong_topic_blocked():
    out = _verify([{"text": "Tumour spheroids resist cisplatin through hypoxia signalling.", "type": "literature",
                    "evidence": ["FIXTURE1::s1::0"]}])
    assert out[0]["status"] == "unsupported"


def test_supported_literature_and_inference_chain():
    out = _verify([
        {"text": "Out-of-focus blur reduces the local variance of the Laplacian [FIXTURE1::s1::0].",
         "type": "literature", "evidence": ["FIXTURE1::s1::0"]},
        {"text": "So re-focusing is advisable.", "type": "inference", "evidence": ["c1"]},
        {"text": "Orphan inference.", "type": "inference", "evidence": []}])
    assert [e["status"] for e in out] == ["supported", "supported", "unsupported"]


def test_ledger_schema_and_extract():
    out = _verify([{"text": f"Level is REVIEW [{TID}]", "type": "inference", "evidence": [TID]}])
    assert not ledger.validate_entry(out[0])
    assert "claim_id" in ledger.LEDGER_SCHEMA["required"]
    p = parse_response('```json\n{"claims":[{"text":"x 0.41 [qc-1]","evidence":["qc-1"]}]}\n```')
    assert p["claims"][0]["type"] == "numeric"
    assert parse_response("ABSTAIN: no evidence")["abstain"]
    assert len(extract_claims("First sentence here. Second one has 4.2 in it.")) == 2


def test_repair_blocks_unfixed(monkeypatch):
    draft = [{"text": "Focus is 0.99.", "type": "numeric", "evidence": [TID]}]
    led = gate.verify(draft, [QC], EV)
    def fake(m):
        return {"text": json.dumps({"claims": [{"text": "Focus is 0.98.", "type": "numeric", "evidence": [TID]}]}),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    r = gate.repair(draft, led, fake, tool_outputs=[QC], evidence=EV)
    assert r["llm_calls"] == 1 and r["ledger"]["n_supported"] == 0 and r["blocked"]
    def good(m):
        return {"text": json.dumps({"claims": [{"text": f"Focus is 0.41 [{TID}].", "type": "numeric", "evidence": [TID]}]}),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    r2 = gate.repair(draft, led, good, tool_outputs=[QC], evidence=EV)
    assert r2["ledger"]["n_supported"] == 1


def test_calc_safe():
    assert calc.safe_eval("(0.41 - 0.30) / 0.30 * 100") == pytest.approx(36.6667, rel=1e-4)
    assert calc.safe_eval("max(1, abs(-3)) ** 2") == 9
    for bad in ["__import__('os').system('id')", "().__class__", "x + 1", "'a'*3", "[i for i in range(3)]",
                "9**9**9", "open('f')", "lambda: 1", "1 if True else 2", ""]:
        with pytest.raises(calc.CalcError):
            calc.safe_eval(bad)
