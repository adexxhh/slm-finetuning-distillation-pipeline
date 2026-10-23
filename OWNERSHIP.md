# Engineering changes and project ownership

## A defensible description
"I implemented a text-to-SQL research pipeline with completion-only LoRA training, real saved-adapter inference, SQL/schema validation and evaluation of independent model predictions. I verified the CPU mechanics on a tiny model. The 8B CUDA QLoRA experiment and its accuracy are still pending."

This describes implemented work without claiming a full-model experiment that cannot currently be reproduced.

## Trace the whole path
1. data_engine/schemas.py defines one domain schema. generator.py creates a small handwritten seed set and reproducible train/validation/test splits.
2. training/formatting.py constructs the prompt, tokenizes the answer separately and masks prompt labels with -100.
3. training/train.py loads a base model, attaches LoRA modules, trains and saves adapter/tokenizer/manifest.
4. serving/backend.py loads real local weights, optionally attaches an adapter, generates tokens and extracts SQL.
5. serving/app.py checks readiness, limits concurrent generation and validates output. It never runs generated SQL.
6. evals/predict.py receives only schema/question, not the reference answer. It writes SQL keyed by the prompt hash.
7. evals/benchmark.py joins independent predictions to references, validates SQL, executes a read-only fixture and records matches/errors.
8. export/quantize.py merges actual adapter weights. GGUF requires an explicitly supplied external toolchain.

## Before and after
| Earlier issue | Repair | Engineering consequence |
|---|---|---|
| Serving always returned fixed SQL | Actual local HF/PEFT model loading; missing model fails 503 | An installed library is no longer confused with a loaded inference engine |
| Benchmark used reference SQL as the model prediction | Independent prediction file with exact IDs | Model output must be generated before accuracy can be computed |
| Baseline scores and latency assigned in code | Withdraw reports; only record observed outputs/timing | No invented frontier-model comparison, speedup or cost saving |
| 80% was over five simulated cases | Withdraw as model-quality evidence | The percentage cannot be defended as generalization |
| Loss trained on prompts/schema too | Completion-only labels | Optimization focuses on the supervised answer, not copying the instruction |
| Validation data also called test data | Separate seeded splits and prompt-overlap checks | Holdout roles are explicit; still only a small single-schema fixture |
| Overlong examples silently truncated answers | Fail overlong examples | A run cannot silently train without meaningful target tokens |
| Training/export errors printed success-like notices | Fail with errors and verify checkpoint presence | Automation cannot claim a successful run when training/export did not occur |
| Backend label inferred from installed packages | Actual backend object/readiness | Health describes initialized resources |
| Claimed TTFT despite buffered mock output | TTFT null; SSE explicitly buffered | Transport shape is separated from actual token streaming |
| SQLite SQL was altered by string substitutions | Execute the submitted SQL unchanged | Dialect incompatibility is surfaced instead of quietly changing the task |

## ML concepts you must explain
**Causal modelling:** next-token prediction, cross-entropy and shifted labels. A label of -100 excludes that position from loss. Explain attention masks versus loss masks and the purpose of EOS.

**LoRA:** W' = W + (alpha/r) B A. For W with dimensions d_out by d_in, A is r by d_in and B is d_out by r. Trainable parameters are r(d_in+d_out), not d_in*d_out. Explain which projections are targeted, rank/capacity trade-offs, alpha scaling and adapter merging.

**QLoRA:** frozen quantized base weights plus trainable adapters. Understand NF4, double quantization, compute dtype versus storage dtype, activations/optimizer memory and gradient checkpointing. Four-bit weights do not make total training memory equal to four bits per parameter. CPU LoRA and CUDA four-bit QLoRA are different experiments.

**Distillation:** standard SFT on handwritten SQL is not automatically knowledge distillation. Explain teacher-generated target supervision versus matching teacher probabilities, provenance and quality filtering. No teacher run is established by the current seed data.

**Evaluation:** syntax validity, schema validity, executable SQL and correct answers are different measures. Execution matches on one DB may be accidental, especially empty results. Understand row order, duplicates, nulls, floating-point comparisons and schema/dialect mismatches. Stronger evaluation needs multiple database instances and unseen schemas.

**Experimental design:** keep validation for tuning and final test for reporting. Include a base-model baseline with identical prompts/decoding, actual predictions, seeds, data hashes, hardware, adapter configuration and failure analysis. Twenty examples with honest results are more useful than a simulated large comparison.

**Inference:** chat templates, prompt/answer boundaries, max_new_tokens, deterministic greedy decoding, model.eval and inference_mode. Explain why an HTTP response chunk is not a newly generated model token.

**Performance:** real TTFT includes prefill until the first generated token. End-to-end time includes queuing, tokenization, inference and validation. Tokens/sec uses tokenizer token counts. A word-count approximation and synthetic sleeps cannot establish model speed.

## Hands-on ownership checkpoints
- Calculate LoRA trainable parameters for one target projection.
- Open a tokenized example and show which labels contribute to loss.
- Change the sequence limit until an example is rejected; explain why this is intentional.
- Run the offline tiny CPU smoke script and inspect saved adapter weights and manifest.
- Reload base+adapter and compare with a merged model.
- Run two unrelated questions through the real backend; explain why a tiny random model's output is not meaningful SQL quality.
- Supply a valid SQL prediction that answers the wrong question and show evaluation rejects it.
- Show an unexecutable reference stays visible in the report.
- Add a reviewed portable subset and a second database fixture before reporting accuracy.
- On CUDA, reproduce an actual 8B run, save artifacts, compare base and tuned outputs and publish failure cases.

## Verified versus pending
15 lightweight tests and actual two-step tiny CPU LoRA training/save/reload/generation passed. Tiny-model adapter merge was also executed. The smoke run has no accuracy claim.
Pending: actual 8B CUDA training, model-quality results, a matched PostgreSQL evaluation fixture, representative holdout data, live token streaming, Docker and external GGUF conversion. These are explicit boundaries, not hidden fallback behaviour.

Start with the LoRA paper https://arxiv.org/abs/2106.09685 and QLoRA paper https://arxiv.org/abs/2305.14314. Be able to explain the equations using your own model shapes and measured runs.
