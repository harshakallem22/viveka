# Viveka

**An LLM evaluation harness** — LLM-as-judge scoring with *validated* reliability, SRE-style
latency metrics (P50/P95, TTFT, throughput), a live leaderboard, and CI regression gating that
fails the build when quality drops.

[![CI](https://github.com/harshakallem22/viveka/actions/workflows/ci.yml/badge.svg)](https://github.com/harshakallem22/viveka/actions/workflows/ci.yml)
[![Regression gate](https://github.com/harshakallem22/viveka/actions/workflows/gate.yml/badge.svg)](https://github.com/harshakallem22/viveka/actions/workflows/gate.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-225%20passing-brightgreen.svg)](#quickstart)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Viveka* is Sanskrit for **discernment** — telling truth from falsehood. The core is an
LLM-as-judge system, so the name is the job.

Most LLM tooling asks *"what can the model do?"* This asks *"how would you know?"*

## Results

3 models × 60 items (GSM8K + TruthfulQA), **180 generations, 0 request failures**, on an M3
MacBook via Ollama.

| model | accuracy | 95% CI | TTFT p50 | lat p50 | lat p95 | tok/s p50 |
|---|---|---|---|---|---|---|
| **gemma3:4b** | **92.0%** | 81–97% | 964 ms | 8.64 s | 22.42 s | **20.2** |
| llama3.1:8b | 82.0% | 69–90% | 787 ms | 10.12 s | 19.81 s | 12.2 |
| mistral:latest | 52.0% | 39–65% | 675 ms | 13.81 s | 27.63 s | 12.5 |

Percentiles are **nearest-rank** — every figure is a value actually observed. Warmup calls are
excluded.

- **The smallest model won on both axes.** gemma3:4b (3.3 GB) was the most accurate *and* the
  fastest; the expected size/quality tradeoff didn't appear.
- **gemma and llama's intervals overlap**, so on 50 items they are *not statistically separable* —
  the leaderboard says so rather than implying a ranking the data can't support.
- **mistral was worst at math but best at resisting falsehoods**, which a single aggregate score
  would have erased.

## Four measurement bugs it caught

Each would have produced a confident, wrong leaderboard. Each is now a regression test.

| Finding | Why it matters |
|---|---|
| A cold call was **6.8× slower** than warm — **93% weight loading**, not inference | With 60 items, one unwarmed call *is* your P95. You'd be reporting SSD read speed. |
| Wall-clock throughput understated decode speed by **89×** (0.50 vs 44.7 tok/s) | Throughput must come from the model's decode counter, not elapsed time. |
| A **90.6 s** worst case was a degenerate repetition loop — 56 lines, 17 unique | The tail was a decoding failure. P50/P95 alone can't show that. |
| A 512-token cap turned a **correct** answer into a wrong one | A truncated response is a corrupted measurement, not a bad model. |

## The judge is validated, not assumed

GSM8K answers are objectively known, so every generation was scored **twice** — once by a
deterministic grader (ground truth), once by an LLM judge (`qwen3:8b`, never a contestant).

| metric | value |
|---|---|
| **Cohen's κ** | **0.982** (almost perfect) |
| raw agreement | 99.3% over 150 pairs |
| **false positives** | **0** — never approved a wrong answer |

κ rather than raw agreement, deliberately: ground truth is 75% correct, so a judge that blindly
replied "correct" would score 75% agreement while being useless. Zero false positives is the
number that matters — that's the direction that inflates a leaderboard *and* lets a regression
pass the gate.

**Limitation:** this validates an *easy* task — number vs number. It does **not** establish
reliability on open-ended claims, which is exactly where the judge is the only grader available.

## CI regression gating

A free GitHub runner can't benchmark an 8B model, so evidence generation is separated from gate
enforcement: the benchmark runs locally, `results/latest.json` is committed, and CI compares it
against the accepted baseline in milliseconds — **exiting non-zero on regression**.

Four tested defences stop gamed evidence: the eval-set hash must match, results must describe the
eval set currently on disk, results must be newer than the prompt code (git timestamps), and
models can't be quietly dropped. **Gating is per-item, not aggregate** — 4 items breaking while 4
improve leaves accuracy flat, and only item-level comparison catches it.

```
│ max_accuracy_drop    │ gemma3:4b │ FAIL │ 92.0% -> 80.0% (drop 12.0%), allowed 10.0%   │
│ max_regressed_items  │ gemma3:4b │ FAIL │ 6 item(s) went correct->incorrect (limit 3)  │

GATE FAILED — 3 blocking check(s)          # exit code 1
```

Every PR also runs the whole pipeline with no GPU: a `ReplayProvider` serves recorded real model
output, so `run → judge → report → gate` executes deterministically in seconds.

## Quickstart

### See it working in 2 minutes — no GPU, no Ollama, no downloads

The repo ships a 400 KB database holding a **real benchmark run**:

```bash
git clone https://github.com/harshakallem22/viveka && cd viveka
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev,api]"

make serve-demo          # builds the dashboard, serves on http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000** for the leaderboard, latency distribution, judge reliability, and a
per-item drill-down — click any row to see the model's full response and every grader's verdict.

Still no models needed:

```bash
make test                # 225 tests
make golden-set-check    # verify the frozen eval set against its hash
./.venv/bin/viveka gate  # run the regression gate against the committed baseline
```

### Run your own benchmark

Needs [Ollama](https://ollama.com) and ~18 GB of models.

```bash
ollama pull llama3.1:8b && ollama pull mistral && ollama pull gemma3:4b && ollama pull qwen3:8b

./.venv/bin/viveka run                           # full benchmark, ~25 min
./.venv/bin/viveka judge latest                  # score it, ~3 min
./.venv/bin/viveka report latest --grader numeric_exact_match
./.venv/bin/viveka validate-judge latest         # judge reliability, ~16 min
make serve                                       # dashboard on :8000
```

`make help` lists every target. Commands are shown as `./.venv/bin/viveka` so they work without
activating the virtualenv.

## Using it as a library

`viveka.core` imports **nothing** from the API, CLI or dashboard — enforced by an AST
import-boundary test and a no-extras install job in CI, not by convention.

```python
from viveka.core import load_golden_set, summarize, percentile, wilson_interval
from viveka.core.providers import OllamaProvider

items = load_golden_set("data/golden_set.jsonl")   # hash-verified
provider = OllamaProvider("gemma3:4b")
provider.warmup()                                   # keeps cold start out of the percentiles
print(provider.complete(items[0].question).timing.decode_tps)
```

The key extension point is the `Provider` protocol — anything turning a prompt into *timed text*.
Implement `complete/warmup/unload` and the whole metrics, judging, storage and gating stack applies
unchanged: a RAG pipeline, a fine-tuned adapter, a hosted API.

## License

MIT — see [LICENSE](LICENSE).
