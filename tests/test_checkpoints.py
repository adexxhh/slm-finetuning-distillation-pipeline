import json
from argparse import Namespace
import pytest
from training.checkpoints import resolve_resume

def options(tmp_path):
    train=tmp_path/"train.jsonl";train.write_text("training data")
    return Namespace(output_dir=str(tmp_path/"output"),model_id="fixed-model-revision",
        train_file=str(train),val_file=None,seed=3407,batch_size=1,gradient_accumulation_steps=8,
        max_seq_length=2048,learning_rate=.0002,max_steps=100,epochs=1,resume_from_checkpoint="auto")

def test_resume_skips_partial_checkpoint_and_rejects_changed_data(tmp_path):
    args=options(tmp_path);assert resolve_resume(args) is None
    out=tmp_path/"output"
    for n in [10,20]:
        p=out/f"checkpoint-{n}";p.mkdir()
        (p/"trainer_state.json").write_text(json.dumps({"global_step":n}))
    p=out/"checkpoint-10"
    for name in ["complete.json","optimizer.pt","scheduler.pt"]:(p/name).write_text("{}")
    assert resolve_resume(args)==str(p)
    (tmp_path/"train.jsonl").write_text("different data")
    with pytest.raises(ValueError,match="configuration/data changed"):resolve_resume(args)

def test_resume_rejects_missing_optimizer_and_completed_run(tmp_path):
    args=options(tmp_path);resolve_resume(args)
    p=tmp_path/"output/checkpoint-100";p.mkdir()
    (p/"trainer_state.json").write_text(json.dumps({"global_step":100}))
    (p/"complete.json").write_text("{}")
    with pytest.raises(ValueError,match="Incomplete"):resolve_resume(args)
    for name in ["optimizer.pt","scheduler.pt"]:(p/name).write_text("{}")
    with pytest.raises(ValueError,match="already reached"):resolve_resume(args)
