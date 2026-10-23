"""Validate experiment identity before restoring optimizer/model checkpoints."""
from pathlib import Path
import hashlib,json

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resolve_resume(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    config=Path(args.model_id)/"config.json"
    identity={"base":file_hash(config) if config.exists() else args.model_id,
              "train":file_hash(args.train_file),
              "validation":file_hash(args.val_file) if args.val_file and Path(args.val_file).exists() else None,
              **{key:getattr(args,key) for key in
                 ["seed","batch_size","gradient_accumulation_steps","max_seq_length","learning_rate","max_steps","epochs"]}}
    requested=args.resume_from_checkpoint
    target=out/"training_identity.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8"))!=identity:
        raise ValueError("Training configuration/data changed. Use a new output directory.")
    if requested=="auto":
        candidates=[p for p in out.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()
                    and (p/"complete.json").exists() and (p/"trainer_state.json").exists()]
        resume=str(max(candidates,key=lambda p:int(p.name.split("-")[-1]))) if candidates else None
    else:resume=requested
    if resume:
        if not target.exists():
            raise ValueError("Checkpoint has no saved training identity; do not auto-resume unverifiable state.")
        if Path(resume).resolve().parent!=out.resolve():
            raise ValueError("Resume checkpoint must belong to the configured output directory.")
        for name in ["trainer_state.json","optimizer.pt","scheduler.pt"]:
            if not (Path(resume)/name).exists():raise ValueError("Incomplete checkpoint: "+name)
        state=json.loads((Path(resume)/"trainer_state.json").read_text(encoding="utf-8"))
        if args.max_steps>0 and state["global_step"]>=args.max_steps:
            raise ValueError("Checkpoint already reached max_steps. Skip training and use its adapter for prediction.")
    target.write_text(json.dumps(identity,indent=2),encoding="utf-8")
    return resume
