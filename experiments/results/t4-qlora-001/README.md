# T4 QLoRA experiment 001

Completed Qwen3-8B NF4 adaptation: 1,000 training questions, 64 validation questions, 100 evaluation questions, 100 optimizer steps on a Tesla T4.

| Model | Original execution matches |
|---|---:|
| Base | 70 / 100 |
| Adapter | 71 / 100 |

Paired outcomes: 61 both match, 20 neither match, 10 adapter-only matches, 9 base-only matches. The comparison does not establish a reliable quality gain on this small cohort.

The directory preserves original predictions, score reports, training state/metrics and provenance. analysis-summary.json and case-analysis.json add explicit column-order and numeric-variation diagnostics. reviewed-cases.json records their interpretation. These diagnostics do not overwrite original scores.

[Experiment report](../../../docs/EXPERIMENT.md) explains methods, results and reproduction. [Data attribution](ATTRIBUTION.md) documents dataset provenance and licensing.

Replay offline with the original results archive:

~~~sh
pip install -r requirements-reproduce.txt
python -m experiments.reproduce --archive /path/to/sql-8b-results.zip --output-dir artifacts/replay
~~~

Or download the 18 pinned SQLite fixtures, without model weights:

~~~sh
python -m experiments.reproduce --download-fixtures --output-dir artifacts/replay
~~~
