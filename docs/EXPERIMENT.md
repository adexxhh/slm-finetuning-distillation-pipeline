# Qwen3-8B SQL adaptation: experiment 001

## Question and outcome

Can a bounded NF4 QLoRA run improve execution matching on database schemas held out from adaptation?

The completed experiment compares the same quantized base model with its trained adapter on a frozen 100-question cohort. Original single-database execution matches are **70/100 for the base and 71/100 for the adapter**. Ten questions improve and nine regress. The observed one-point difference does not establish a reliable improvement.

## Configuration

| Item | Recorded value |
|---|---|
| Model | Qwen/Qwen3-8B |
| Model revision | b968826d9c46dd6066d109eabc6255188de91218 |
| Hardware | Tesla T4, 14.56 GiB reported GPU memory |
| Adaptation | NF4, double quantization, LoRA rank 16 / alpha 16 |
| Targets | q/k/v/o projections and gate/up/down projections |
| Adapter parameters | 43,646,976; saved as FP32 |
| Training examples / databases | 1,000 / 126 |
| Validation examples / databases | 64 / 13 |
| Evaluation examples / databases | 100 / 18 |
| Microbatch / accumulation | 1 / 8 |
| Sequence length / learning rate | 2,048 / 0.0002 |
| Seed / optimizer steps | 3407 / 100 |
| Completed epochs | 0.8 |
| Reported Trainer runtime | 5,117.75 seconds, about 85.3 minutes |
| Final recorded validation loss | 0.26359 |
| Saved checkpoint evidence | Steps 90 and 100, including full optimizer state |

Training uses SQL-answer-only causal loss. The base and adapter predictions use the same schema/question context, greedy decoding and maximum 256 generated tokens. No prediction reached that token cap.

The GPU experiment was run once. Repeating training across seeds or comparing alternative ranks/model sizes was outside this bounded experiment.

## Data and provenance

Training, validation and evaluation database IDs are disjoint; evaluation prediction IDs match the frozen questions exactly. Data and database revisions, split hashes and fixture hashes are preserved in the evidence directory.

Question revision: 0c350918f3f29ec754f1181c65cdce76cd6c133c.
Database mirror revision: 8698f764f0547eb7136b3be9569093c7f4217be4.

Preparation filters unsupported, non-executable or overlong reference examples. The manifest records 279 skipped candidates before collecting the training subset, 11 before validation and 27 before evaluation. This is a restricted eligible cohort, not all Spider data. Public benchmark contamination in model pretraining has not been ruled out.

Packaging normalizes database paths. The manifest preserves hashes before and after that normalization; training_identity.json matches the pre-packaging training hashes. Model inputs are the stored conversation text; the database path field is evaluation metadata.

The frozen evidence is at experiments/results/t4-qlora-001. Its evidence-lock.json verifies 20 saved raw/analysis files. The original 1.19 GB results archive stays outside Git; its SHA256 is:
3f6634b99a05f8ce7b2c530c22cc9c89da94c4967af6c5ea1fdff9026e8f6c9c.

## Result definitions

The primary metric keeps output columns in reference order. If the reference contains ORDER BY, row order is compared; otherwise row multisets are compared, preserving duplicate counts. Projection width is checked, including empty results. Missing/duplicate question IDs and empty evaluations fail validation.

Queries execute with read-only SQLite access, deadlines and a 10,000-row result budget. Syntax/schema rejection, execution error and output mismatch remain distinct failure categories.

A column-permutation diagnostic finds one consistent reordering across an entire result, for up to four columns. It does not sort individual rows' values or silently promote these cases into the primary score. This avoids treating unspecified natural-language column order as a hidden scoring choice.

## Results

| Measure | Base | Adapter |
|---|---:|---:|
| Original execution matches | 70 / 100 | 71 / 100 |
| Syntax/schema policy accepted | 94 / 100 | 89 / 100 |
| Column-order-only mismatches | 3 | 2 |
| Recorded mean generation time | 3.93 s | 5.15 s |
| Recorded median generation time | 3.47 s | 4.00 s |

Generation timing covers the recorded model generation calls; it excludes model loading and is a single-run descriptive measurement.

| Paired outcome | Questions |
|---|---:|
| Both match | 61 |
| Adapter only | 10 |
| Base only | 9 |
| Neither matches | 20 |

An exploratory 5,000-draw database-cluster bootstrap gives a score-difference interval of approximately **-10.74 to +12.09 percentage points**. This acknowledges database-level clustering, but 18 clusters and a filtered cohort still limit inference. Original reports also preserve per-question Wilson intervals; those do not account for clustering.

## Semantic sensitivity checks

Three deterministic database variants per evaluation database shuffle bounded non-key numeric values. The same generator runs for both models. It preserves row identities/counts, NULL positions, declared keys and text values, and rolls back updates that violate constraints.

The generator touches at most 500 rows per table and uses seeds 3407, 3408 and 3409. Sixteen databases have eligible numeric changes; two have none. It is a deliberately limited sensitivity check, not the official Spider test-suite evaluator or a proof of equivalence.

Among original matches, 63 base cases and 61 adapter cases receive an eligible numeric check. One original match from each model fails at least one variant:

- Base, poker_player: selecting the tallest person before checking whether that person is a poker player can disagree with selecting the tallest poker player.
- Adapter, dog_kennels: ranking owners by total dog weight can disagree with ranking them by treatment cost.

The adapter example fails all three generated variants. Both are concrete counterexamples to semantic equivalence, even though the original fixture produced a matching answer. Their original scores remain preserved so the historical metric can be replayed exactly.

Ties, floating-point behavior and reference ambiguity can also cause mismatches under changed data. Each flagged case requires inspection. A passing variant check does not certify the remaining queries.

## Error analysis and interpretation

The adapter produces improvements in some set operations, grouping and filtering cases. Its regressions include referencing columns from the wrong table, omitted DISTINCT operations and result-shape differences. Schema-policy acceptance falls from 94 to 89 cases.

The decreasing training loss and final validation loss establish optimization behavior; they do not establish a quality gain over the base model. The evidence supports a completed resource-constrained adaptation experiment with a small, uncertain net difference.

Future experiments should use validation to choose schema-context, learning-rate or data changes. This inspected evaluation cohort must not become the tuning set for a new final claim; retain a fresh holdout for the next iteration.

## Reproduce evaluation without a GPU

From the repository root, create a fresh Python environment and install only the pinned evaluator dependency:

~~~sh
python -m venv .venv-replay
# Activate it, then:
pip install -r requirements-reproduce.txt
~~~

Offline, using the downloaded archive:

~~~sh
python -m experiments.reproduce --archive /path/to/sql-8b-results.zip --output-dir artifacts/replay
~~~

With public fixture downloads:

~~~sh
python -m experiments.reproduce --download-fixtures --output-dir artifacts/replay
~~~

The second command downloads 18 pinned SQLite fixtures (about 106 MB uncompressed) and no model weights. Both commands verify evidence hashes, disjoint splits, prediction IDs and the frozen scores, then produce summary.json, case-analysis.json and numeric-variants.json. The default diagnostic settings are checked against the recorded model summaries and paired analysis.

To use a pre-existing fixture directory, pass --database-dir /path/to/databases instead of an archive/download flag. For a scoring-only replay, use --variants 0; the original score checks still run.

To run the standard scorer with portable paths:

~~~sh
python -m experiments.spider score --test-file experiments/results/t4-qlora-001/spider/test.jsonl --predictions experiments/results/t4-qlora-001/adapter-predictions.jsonl --database-dir artifacts/replay/databases --output artifacts/adapter-rescored.json
~~~

## Repeat training configuration

Download the base snapshot using the recorded model revision and match the package versions in environment.json when attempting a GPU rerun. Use the frozen training and validation conversations with a new output directory:

~~~sh
python -m training.train --model_id /path/to/pinned/base --train_file experiments/results/t4-qlora-001/spider/train.jsonl --val_file experiments/results/t4-qlora-001/spider/validation.jsonl --output_dir artifacts/new-training-run --batch_size 1 --gradient_accumulation_steps 8 --max_seq_length 2048 --learning_rate 0.0002 --max_steps 100 --epochs 1 --seed 3407 --save_steps 10
~~~

A full second GPU training run has not been executed, so exact adapter-weight or generation equality across GPU/software environments is not claimed. CPU evaluation replay is the verified reproduction target.

For preparation from upstream sources, experiments.spider prepare accepts --question-revision, --database-revision and --seed. Use the revisions above and the recorded limits. Frozen conversations avoid changes in dataset eligibility from newer tokenizer/parser versions.

## Artifacts and references

[Saved evidence](../experiments/results/t4-qlora-001/README.md) contains predictions, metrics, per-case diagnostics and training/provenance metadata. [Data attribution](../experiments/results/t4-qlora-001/ATTRIBUTION.md) identifies the dataset license and sources. Weights and binary optimizer state remain in the original local archive.

The official semantic evaluation project is https://github.com/taoyds/test-suite-sql-eval. Its metric and generated database suites are distinct from the bounded sensitivity checks implemented here.

## Reproduction verification

A fresh source directory and new Python 3.13.5 environment containing only sqlglot 30.24.0 replayed the original archive, replayed an independently resolved fixture directory and ran the standard scorer. Both scores matched. The three diagnostic JSON outputs were byte-identical across repeated local runs. [Verification record](evidence/cpu-replay-verification.json) preserves the environment and output hashes. Frozen evidence uses explicit LF line endings so byte checksums survive cross-platform checkouts.
