# Text-to-SQL LoRA Research Pipeline

A research prototype connecting supervised fine-tuning, real local inference, AST/schema checks and independent SQL evaluation.

**Evidence:** tiny-model CPU LoRA training/inference works. **No validated 8B QLoRA accuracy result exists.** Previous accuracy, model comparisons, latency and cost tables were simulated and are withdrawn.

~~~mermaid
flowchart LR
  Data[Seed data] --> Split[Train / validation / test]
  Split --> Train[Completion-only LoRA]
  Train --> Adapter[Saved adapter]
  Adapter --> Model[Real local HF model]
  Model --> Predictions[Independent predictions]
  Predictions --> Evaluation[Read-only SQL evaluation]
~~~

## CPU checks
~~~sh
pip install -r requirements.txt
python -m pytest -q
python main.py --generate
python main.py --validate data/train.jsonl
python main.py --validate data/validation.jsonl
python main.py --validate data/test.jsonl
~~~
The dataset is a small handwritten single-schema fixture. Splits are seeded and disjoint by prompt hash. This is not evidence of teacher-model distillation or cross-schema generalization. Parsing does not prove that SQL correctly answers an English question.

## Actual training smoke run
~~~sh
pip install -r requirements-ml.txt
python -m training.smoke_cpu
~~~
Creates a tiny random Llama/tokenizer locally, performs two real CPU LoRA optimizer steps, saves/reloads the adapter and generates tokens. No pretrained model download. **Not 8B training, 4-bit QLoRA or a quality benchmark.**
The run passed along with **15 lightweight tests**. [Recorded CPU evidence](docs/evidence/cpu-smoke.json).

## CUDA QLoRA (unverified here)
Install a compatible CUDA PyTorch/bitsandbytes stack and obtain an authorized model:
~~~sh
python -m training.train --model_id /models/base --train_file data/train.jsonl --val_file data/validation.jsonl --output_dir artifacts/adapter
~~~
No CUDA: the command fails unless --allow_cpu explicitly selects full-precision LoRA for a small model. Overlapping train/validation prompts and overlong examples fail. Training emits a manifest.
An adequate dataset, hardware budget and independent full-model evaluation remain necessary.

## Serve actual weights
Set MODEL_PATH to a downloaded local model directory, optionally ADAPTER_PATH to an adapter, then run uvicorn serving.app:app --host 127.0.0.1 --port 8000.
- /health reports whether a model was actually loaded.
- /predict/sql generates and validates SQL but never executes it.
- Missing model returns HTTP 503; rejected output returns HTTP 422.
- /v1/chat/completions supports one user question. SSE is explicitly **buffered**, not live token streaming. TTFT is null instead of fabricated.
- Schema is controlled by the server; arbitrary custom schemas are not supported.
Docker uses tiny CPU weights as an example and was not executed here.

## Independent evaluation
~~~sh
python -m evals.predict --model /models/base --adapter artifacts/adapter --test_file data/test.jsonl --output artifacts/predictions.jsonl
python -m evals.benchmark --test_file data/test.jsonl --predictions artifacts/predictions.jsonl --output_dir artifacts/evaluation
~~~
Requires exact prompt IDs and independent predictions; reference answers never enter inference. The deadline-limited read-only SQLite fixture compares ordered results when ORDER BY exists, otherwise row multisets. Invalid references remain visible in the denominator.
**PostgreSQL-specific queries can fail on SQLite.** Use a matched PostgreSQL fixture or a reviewed portable subset. One small DB can produce coincidental result matches; multiple data instances are needed for strong semantic evidence.

## Export actual weights
~~~sh
python -m export.quantize --base_model_id artifacts/cpu_smoke/base --lora_path artifacts/cpu_smoke/adapter --output_dir artifacts/exported --format merged
~~~
Tiny-model merging was executed. CPU merge uses FP32. GGUF additionally requires --format gguf --llama_cpp_dir with a converter and built quantizer; it was not verified here. AWQ is no longer advertised as an automatic implementation.


## Demonstration
![Verified terminal demonstration](assets/demo.gif)

[Static PNG](assets/demo.png) | [Raw commands and output](assets/demo-transcript.json)

These assets are condensed terminal replays rendered from actual subprocess output. They are not screenshots of a production deployment.
Regenerate: install requirements-demo.txt, then run python scripts/render_demo.py.

[Engineering changes and ownership exercises](OWNERSHIP.md)

## Actual GPU experiment

[Executable Colab/Kaggle notebook](experiments/colab_8b_qlora.ipynb) prepares pinned Spider data, trains Qwen3-8B with NF4 QLoRA, and compares independent base/adapter predictions on held-out databases. It records confidence intervals, failures, revisions and artifacts. GPU execution remains pending; no 8B accuracy is claimed. See [experiment scope](experiments/README.md).
