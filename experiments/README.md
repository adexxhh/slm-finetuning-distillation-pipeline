# GPU training and evaluation

[Training notebook](colab_8b_qlora.ipynb) configures Qwen3-8B NF4 QLoRA with Spider data and compares base-model and adapter predictions.

Each Markdown block describes the next cell's process and expected output.

## Run

1. Upload the notebook to Colab and select a CUDA GPU runtime.
2. Upload the matching sql-8b-source.zip when prompted.
3. Run cells in order and complete Google Drive authorization at the training cell.
4. Download sql-8b-results.zip after evaluation.

Build source archives from clean sibling project checkouts with python scripts/build_source_bundles.py --output-dir ../ from the SQL repository.

## Checkpoint recovery

Training writes progress logs and full checkpoints every 10 optimizer steps to MyDrive/ai-project-experiments/sql-8b-run-001/adapter. Keep that directory for recovery. Auto-resume validates dataset/configuration fingerprints and selects a checkpoint marked complete.

Use a new output directory when changing the dataset or training configuration. Model downloads and dataset preparation may need to run again after a runtime reset.

## Outputs

The result archive contains model revision and package versions, split manifests, prediction records, execution-match reports, SQLite fixtures and adapter state. Inspect validation loss, base/adapter scores and failure cases together.

Evaluation uses a database-disjoint public Spider development subset and single-database execution matching. It is a different metric from official Spider test-suite accuracy.

## Separate code-repair experiment

The source bundle also contains the editor's repair probes. Run those in a separate disposable session without Google Drive mounted.

## Completed run and replay

[Experiment 001](../docs/EXPERIMENT.md) records the completed T4 run, saved predictions and analysis. Run python -m experiments.reproduce --download-fixtures --output-dir artifacts/replay after installing requirements-reproduce.txt to replay evaluation without a GPU. Offline replay accepts --archive with the original results ZIP.
