# GPU experiment
Open colab_8b_qlora.ipynb and upload the matching sql-8b-source.zip. Each Markdown block describes the cell below it and its expected output.

The notebook streams subprocess output and records logs. Training saves full checkpoints every 10 optimizer steps to MyDrive/ai-project-experiments/sql-8b-run-001/adapter after you authorize the Google Drive mount. Keep that directory for recovery. Use a new directory if you change the dataset or training configuration. Resume validates input/configuration fingerprints and skips checkpoints without a completion marker.

Local checks: 17 tests pass. An interrupted tiny CPU training run restored optimizer/scheduler state at step 2 and continued through step 4 with changed adapter weights. The CUDA/Drive workflow has not been verified end to end.

Validation uses batch size one and loss-only output to avoid retaining full vocabulary logits. Baseline and adapter predictions are independently generated and scored on the same populated, held-out databases. Single-database execution matching is not official Spider test-suite accuracy.

Model-driven repair probes remain in the source bundle under repair-editor/benchmarks. Run them in a separate disposable session without personal Drive mounted.
