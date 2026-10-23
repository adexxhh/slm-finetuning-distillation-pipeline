# Full-model experiment
Use the GPU notebook in the top-level repair bundle. It prepares a seeded Spider split by database, records dataset revisions and file hashes, trains an actual 8B QLoRA adapter and evaluates base versus adapter on the same public development subset.

The SQLite files contain actual rows. Independent prediction IDs and raw SQL are saved. Accuracy is single-instance execution matching, not official Spider test-suite accuracy; empty-reference matches and a Wilson interval are reported separately.
No run has been claimed complete until the GPU notebook finishes and exports its artifacts.

Sources:
- https://huggingface.co/datasets/xlangai/spider (original question/SQL release; CC BY-SA 4.0)
- https://huggingface.co/datasets/prem-research/spider (database mirror; record exact revision)
- https://github.com/taoyds/spider (original project)
- https://huggingface.co/Qwen/Qwen3-8B (model card/license)


Start with colab_8b_qlora.ipynb and the matching sql-8b-source.zip. Run cells in order on a CUDA GPU runtime. Neither GPU execution nor 8B accuracy has been verified locally. Download results before session expiry.

The matching source archive also includes repair-editor with three actual-model repair probes. The notebook runs these using the same downloaded base model; results are integration evidence, not a representative success-rate estimate.
