"""Offline tests for hybrid retrieval and the validated single-call judge metrics."""
import json

from deepeval.test_case import LLMTestCase

from evallab.chunking import load_corpus
from evallab.config import settings
from evallab.custom_metrics import make_relevance_metric, make_validated_faithfulness_metric
from evallab.lexical import BM25, rrf, tokenize
from evallab.llm import FakeLLM

G012 = "A Business customer stores 10,000 GB for a month. How much will they pay for storage?"


def test_tokenize_keeps_numbers_and_drops_stopwords():
    assert tokenize("How much is $0.018 per GB-month?") == ["0.018", "gb", "month"]


def test_bm25_finds_the_named_plan_that_dense_retrieval_missed():
    chunks = load_corpus(settings.corpus_dir)
    bm25 = BM25([c.text for c in chunks])
    top = chunks[bm25.rank(G012, 1)[0]]
    assert top.text.startswith("[Helios Cloud Storage: Pricing Plans > Business]")


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a" and set(fused) == {"a", "b", "c", "d"}


def test_hybrid_pipeline_returns_top_k_unique_chunks(offline_pipeline):
    res = offline_pipeline.answer(G012)
    assert len(res.contexts) == len(set(res.contexts)) <= offline_pipeline.top_k
    assert any("Pricing Plans > Business" in c for c in res.contexts)


def _case(answer="Business costs $0.018 per GB-month [1].", ctx=("The Business plan costs $0.018.",)):
    return LLMTestCase(input="How much does Business cost?", actual_output=answer,
                       retrieval_context=list(ctx))


def test_validated_faithfulness_metric_passes_and_fails_on_the_verdict():
    ok = make_validated_faithfulness_metric(FakeLLM(responses=[json.dumps(
        {"reason": "supported", "faithful": True})]))
    bad = make_validated_faithfulness_metric(FakeLLM(responses=[json.dumps(
        {"reason": "price contradicted", "faithful": False})]))
    assert ok.measure(_case()) == 1.0 and ok.is_successful()
    assert bad.measure(_case()) == 0.0 and not bad.is_successful()


def test_unparseable_judge_output_fails_loudly():
    m = make_validated_faithfulness_metric(FakeLLM(responses=["not json at all"]))
    assert m.measure(_case()) == 0.0 and not m.is_successful()
    assert "unparseable" in m.reason


def test_relevance_metric_reads_its_own_field():
    m = make_relevance_metric(FakeLLM(responses=[json.dumps(
        {"reason": "answers the price question", "addresses_question": True})]))
    assert m.measure(_case()) == 1.0
