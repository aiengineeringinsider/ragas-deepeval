PY ?= python
RESP = results/responses_qwen2.5_3b.jsonl

.PHONY: setup models check ingest ask golden ragas deepeval judges bias mm gate full test test-eval api dash

setup:
	$(PY) -m pip install -r requirements.txt

models:
	ollama pull qwen2.5:3b && ollama pull qwen2.5vl:3b && ollama pull qwen3:1.7b \
	  && ollama pull llama3.2:1b && ollama pull nomic-embed-text

check:    ; $(PY) scripts/01_check_env.py
ingest:   ; $(PY) scripts/02_ingest.py --reset
ask:      ; $(PY) scripts/03_ask.py "$(Q)"
golden:   ; $(PY) scripts/04_golden.py validate && $(PY) scripts/04_golden.py run --out $(RESP)
ragas:    ; $(PY) scripts/05_ragas_eval.py --responses $(RESP)
deepeval: ; $(PY) scripts/06_deepeval_eval.py --responses $(RESP)
judges:   ; $(PY) scripts/07_compare_judges.py --repeats 3
bias:     ; $(PY) scripts/08_bias_probes.py
mm:       ; $(PY) scripts/make_images.py && $(PY) scripts/09_multimodal_eval.py
full:     ; $(PY) scripts/11_full_run.py --label manual
test:     ; $(PY) -m pytest -m "not ollama" -q
test-eval:; deepeval test run tests/eval/test_rag_quality.py
api:      ; uvicorn api.main:app --reload --port 8000
dash:     ; streamlit run dashboard/app.py

# Part V
testset:  ; $(PY) scripts/12_testset.py --size 12
traced:   ; $(PY) scripts/13_traced_eval.py
convo:    ; $(PY) scripts/14_conversations.py
agent:    ; $(PY) scripts/15_agent_eval.py
monitor:  ; $(PY) scripts/16_monitor.py --records $(RESP)
redteam:  ; $(PY) scripts/17_redteam.py
