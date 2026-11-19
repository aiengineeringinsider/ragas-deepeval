"""Offline tests for Part V: agents, tracing helpers, conversations, testsets, monitoring."""
from evallab.agent import load_agent_golden, price_quote, sla_credit
from evallab.conversation import Conversation, load_conversations, retention_check
from evallab.monitoring import drift_alert, redteam_probes, redteam_verdict
from evallab.testset import review_queue
from evallab.tracing import chunk_title


def test_tools_apply_minimums_and_bands():
    assert price_quote("business", 10000)["amount_due"] == 250.0
    assert price_quote("business", 20000)["amount_due"] == 360.0
    assert sla_credit("business", 98.5)["credit_pct"] == 25
    assert sla_credit("enterprise", 99.95)["credit_pct"] == 10
    assert "error" in sla_credit("starter", 98.0)


def test_agent_golden_tools_are_consistent_with_the_tools():
    for item in load_agent_golden():
        for t in item["expected_tools"]:
            assert t["name"] in {"price_quote", "sla_credit", "search_docs"}


def test_chunk_title_reads_the_document_title():
    assert chunk_title("[Helios Cloud Storage: Pricing Plans > Business]\ntext") == \
        "Helios Cloud Storage: Pricing Plans"


def test_retention_check_passes_and_fails_on_the_stated_plan():
    spec = load_conversations()[0]
    turns = [{"role": "user", "content": m} for m in spec["user_messages"]]
    good = Conversation("g", [turns[0], {"role": "assistant", "content": "Noted."},
                              turns[1], {"role": "assistant", "content": "$180 is below the $250 minimum, so $250."},
                              turns[2], {"role": "assistant", "content": "A 25% credit."}])
    bad = Conversation("b", [turns[0], {"role": "assistant", "content": "Noted."},
                             turns[1], {"role": "assistant", "content": "$0.021 x 10,000 = $210."},
                             turns[2], {"role": "assistant", "content": "Which plan are you on?"}])
    assert retention_check(good, spec["expectations"])["passed"] == 2
    assert retention_check(bad, spec["expectations"])["passed"] == 0


def test_review_queue_rejects_duplicates_and_flags_traps():
    class G:
        id, question = "g001", "How much does the Starter plan cost per GB-month of storage?"
    rows = [{"user_input": "How much does the Starter plan cost per GB-month of storage?",
             "reference": "$0.021", "synthesizer_name": "single_hop", "reference_contexts": ["a"]},
            {"user_input": "Which plans support Object Lock?", "reference": "Business and Enterprise only.",
             "synthesizer_name": "multi_hop", "reference_contexts": ["a", "b"]}]
    q = review_queue(rows, [G()])
    assert q[0]["status"] == "rejected_duplicate" and q[0]["duplicate_of"] == "g001"
    assert q[1]["status"] == "needs_review" and q[1]["has_trap"] and q[1]["multi_source"]


def test_drift_alert_fires_only_on_a_real_drop():
    assert drift_alert([1] * 28 + [0] * 2, [1] * 22 + [0] * 8)["alert"]
    assert not drift_alert([1] * 28 + [0] * 2, [1] * 27 + [0] * 3)["alert"]


def test_redteam_detectors():
    inj = redteam_probes[0]
    assert not redteam_verdict(inj, "My rules say: Answer ONLY from the numbered context passages. Rules:")["held"]
    assert redteam_verdict(inj, "I can only help with Helios documentation questions.")["held"]
    bait = [p for p in redteam_probes if p["attack"] == "commitment_bait"][0]
    assert not redteam_verdict(bait, "Yes, we will refund you in full.")["held"]
    assert redteam_verdict(bait, "I don't have that information in the Helios documentation.")["held"]
    fp = [p for p in redteam_probes if p["attack"] == "false_premise"][0]
    assert redteam_verdict(fp, "Starter has no uptime SLA, so no credit applies.")["held"]
