"""Replay a frozen QLoRA experiment on CPU; no torch, weights or GPU required."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sqlite3,statistics,sys,time,urllib.error,urllib.request,zipfile
import sqlglot
from data_engine.validator import SQLValidator
from evals.benchmark import record_id
from evals.diagnostics import compare_results,numeric_variant,paired_summary,run_query
from experiments.spider import schema_from_db

DEFAULT_EVIDENCE=Path(__file__).resolve().parent/"results/t4-qlora-001"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def verified_inputs(evidence):
    lock=read_json(evidence/"evidence-lock.json")
    if sqlglot.__version__!=lock["sqlglot"]:
        raise ValueError("Use requirements-reproduce.txt; evaluator requires sqlglot "+lock["sqlglot"])
    for relative,digest in lock["frozen_files_sha256"].items():
        path=(evidence/relative).resolve()
        if not path.is_relative_to(evidence.resolve()):raise ValueError("Unsafe evidence path")
        if sha256(path)!=digest:raise ValueError("Frozen evidence hash mismatch: "+relative)
    manifest=read_json(evidence/"spider/manifest.json")
    ids={};dbs={};records={}
    for split in ["train","validation","test"]:
        path=evidence/f"spider/{split}.jsonl"
        if sha256(path)!=manifest["files_sha256"][split]:raise ValueError("Split hash mismatch")
        records[split]=read_lines(path)
        if not records[split]:raise ValueError("Empty split: "+split)
        ids[split]={record_id(r) for r in records[split]}
        if len(ids[split])!=len(records[split]):raise ValueError("Duplicate question IDs in "+split)
        dbs[split]={r["db_id"] for r in records[split]}
        if len(records[split])!=manifest["splits"][split]["count"]:raise ValueError("Split count mismatch")
    for left,right in [("train","validation"),("train","test"),("validation","test")]:
        if dbs[left]&dbs[right] or ids[left]&ids[right]:raise ValueError("Split overlap")
    identity=read_json(evidence/"adapter/training_identity.json")
    for split in ["train","validation"]:
        if identity[split]!=manifest["original_files_sha256"][split]:raise ValueError("Training input provenance mismatch")
    predictions={}
    for label in ["base","adapter"]:
        items=read_lines(evidence/f"{label}-predictions.jsonl")
        predictions[label]={r["id"]:r for r in items}
        if len(predictions[label])!=len(items) or set(predictions[label])!=ids["test"]:
            raise ValueError("Prediction ID mismatch: "+label)
    return lock,manifest,records,predictions

def materialize_databases(manifest,records,destination,archive=None,database_dir=None,download=False):
    destination.mkdir(parents=True,exist_ok=True);paths={}
    source=zipfile.ZipFile(archive) if archive else None
    try:
        for db_id in sorted({r["db_id"] for r in records}):
            if Path(db_id).name!=db_id or "/" in db_id or "\\" in db_id:raise ValueError("Unsafe database ID")
            name=db_id+".sqlite";expected=manifest["database_files_sha256"][name]
            if database_dir:
                target=Path(database_dir)/name
                if not target.is_file() or sha256(target)!=expected:raise ValueError("Missing/mismatched database: "+name)
                paths[db_id]=target;continue
            target=destination/name
            if target.exists():
                if sha256(target)!=expected:raise ValueError("Existing fixture hash mismatch: "+name)
                paths[db_id]=target;continue
            temporary=target.with_suffix(".partial")
            print("Preparing fixture:",db_id,flush=True)
            if source:
                with source.open("artifacts/spider/databases/"+name) as inp,temporary.open("wb") as out:
                    for chunk in iter(lambda:inp.read(1024*1024),b""):out.write(chunk)
            elif download:
                revision=manifest["database_revision"]
                url=f"https://huggingface.co/datasets/prem-research/spider/resolve/{revision}/database/{db_id}/{name}"
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(url,timeout=60) as inp,temporary.open("wb") as out:
                            for chunk in iter(lambda:inp.read(1024*1024),b""):out.write(chunk)
                        break
                    except (urllib.error.URLError,TimeoutError) as exc:
                        if attempt==2:raise
                        print(f"Fixture transfer retry {attempt+1}/2 for {db_id}: {exc}",flush=True)
                        time.sleep(attempt+1)
            else:raise ValueError("Choose --archive, --database-dir or --download-fixtures")
            if sha256(temporary)!=expected:raise ValueError("Downloaded/extracted fixture hash mismatch: "+name)
            temporary.replace(target);paths[db_id]=target
    finally:
        if source:source.close()
    return paths

def describe_errors(query,reference,valid,error):
    if not valid:return "schema_or_policy_rejection"
    if query.error:return "execution_error"
    if query.width!=reference.width:return "output_shape"
    if error["column_permutation_only"]:return "column_order_only"
    return "result_mismatch"

def run(args):
    evidence=Path(args.evidence).resolve();output=Path(args.output_dir).resolve()
    if output.is_relative_to(evidence):raise ValueError("Output must not overwrite frozen evidence")
    lock,manifest,records,predictions=verified_inputs(evidence)
    if args.variants<0 or args.variants>10:raise ValueError("Use zero to ten diagnostic variants")
    if args.archive and sha256(args.archive)!=lock["source_archive_sha256"]:
        raise ValueError("Archive checksum differs from the original experiment")
    output.mkdir(parents=True,exist_ok=True)
    paths=materialize_databases(manifest,records["test"],output/"databases",
        archive=args.archive,database_dir=args.database_dir,download=args.download_fixtures)
    expected={label:{x["id"]:x for x in read_json(evidence/f"{label}-score.json")["details"]} for label in predictions}
    details=[];schemas={}
    for r in records["test"]:
        rid=record_id(r);db=r["db_id"];path=paths[db]
        if db not in schemas:schemas[db]=schema_from_db(path)[0]
        connection=sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            sql=next(x["value"] for x in r["conversations"] if x["from"]=="gpt")
            reference=run_query(connection,sql)
            if reference.error:raise ValueError("Reference is not executable: "+rid+": "+reference.error)
            detail={"id":rid,"db_id":db,"question":next(x["value"] for x in r["conversations"] if x["from"]=="human"),
                    "reference_sql":sql,"reference_rows":len(reference.rows),"reference_width":reference.width,
                    "reference_ordered":reference.ordered}
            for label,pred in predictions.items():
                valid,error=SQLValidator(schemas[db]).validate(pred[rid]["sql"])
                query=run_query(connection,pred[rid]["sql"]) if valid else None
                result=compare_results(reference,query) if query else {"strict":False,"column_permutation_only":False,"mapping":None}
                if valid!=expected[label][rid]["valid"] or result["strict"]!=expected[label][rid]["correct"]:
                    raise ValueError("Replay differs from frozen score: "+label+"/"+rid)
                detail[label]={**result,"sql":pred[rid]["sql"],"valid":valid,"error":error or (query.error if query else None),
                    "category":"matched" if result["strict"] else describe_errors(query,reference,valid,result),
                    "numeric_variant_failures":[],"numeric_variants_checked":0,"numeric_reference_errors":0}
            details.append(detail)
        finally:connection.close()
    variants_meta=[]
    for db,path in sorted(paths.items()):
        cases=[d for d in details if d["db_id"]==db]
        for index in range(args.variants):
            seed=args.seed+index
            source=sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
            try:variant,metadata=numeric_variant(source,seed,args.row_cap)
            finally:source.close()
            print(f"Diagnostic database={db} variant={index+1}/{args.variants} mutated_columns={len(metadata['changes'])}",flush=True)
            variants_meta.append({"db_id":db,"variant":index,**metadata})
            try:
                if not metadata["changes"]:continue
                for d in cases:
                    reference=run_query(variant,d["reference_sql"])
                    for label in predictions:
                        result=d[label]
                        if not result["strict"]:continue
                        if reference.error:
                            result["numeric_reference_errors"]+=1;continue
                        result["numeric_variants_checked"]+=1
                        prediction=run_query(variant,result["sql"])
                        compared=compare_results(reference,prediction)
                        if not compared["strict"]:
                            result["numeric_variant_failures"].append({"variant":index,"seed":seed,
                                "column_permutation_only":compared["column_permutation_only"],
                                "prediction_error":prediction.error,
                                "reference_sample":reference.rows[:3],"prediction_sample":prediction.rows[:3]})
            finally:variant.close()
    summary={"scope":"Frozen 100-question filtered Spider public-development cohort; original single-database execution metric retained",
        "official_spider_test_suite_accuracy":False,"samples":len(details),"database_clusters":len(paths),
        "sqlglot":sqlglot.__version__,"sqlite":sqlite3.sqlite_version,
        "evaluation_code_sha256":{name:sha256(Path(__file__).resolve().parents[1]/name)
            for name in ["evals/diagnostics.py","experiments/reproduce.py","data_engine/validator.py"]},"source_archive_sha256":lock["source_archive_sha256"],
        "verified_frozen_files":len(lock["frozen_files_sha256"]),"verified_databases":len(paths),
        "split_counts":{s:len(r) for s,r in records.items()},"split_database_counts":{s:len({r["db_id"] for r in rs}) for s,rs in records.items()},
        "paired":paired_summary(details,args.seed),"models":{},
        "diagnostics":{"numeric_variants_per_database":args.variants,"seed":args.seed,"row_cap":args.row_cap,
            "column_policy":"Strict positions remain primary; one global permutation up to four columns is a diagnostic only.",
            "numeric_policy":"Seeded bounded shuffles of non-key numeric columns on in-memory copies; preserves declared key values, row counts and NULL positions. Reports sensitivity, not semantic correctness.",
            "limitations":"Variants are post-hoc diagnostics, not a new held-out score. Ties, floating-point behavior and reference ambiguity can cause mismatches. No finite test set proves equivalence."}}
    for label,preds in predictions.items():
        stats=[d[label] for d in details]
        lat=[x["latency_ms"]/1000 for x in preds.values()]
        summary["models"][label]={"strict_matches":sum(x["strict"] for x in stats),
            "strict_execution_match_pct":100*sum(x["strict"] for x in stats)/len(stats),
            "schema_policy_valid":sum(x["valid"] for x in stats),
            "column_order_only_mismatches":sum(x["column_permutation_only"] for x in stats),
            "numeric_variants_checked_for_original_matches":sum(x["numeric_variants_checked"]>0 for x in stats),
            "original_matches_sensitive_to_numeric_variants":sum(bool(x["numeric_variant_failures"]) for x in stats),
            "failure_categories":dict(Counter(x["category"] for x in stats)),
            "recorded_mean_generation_seconds":statistics.mean(lat),
            "recorded_median_generation_seconds":statistics.median(lat)}
    reference_analysis=evidence/"analysis-summary.json"
    if reference_analysis.exists():
        frozen=read_json(reference_analysis)
        same_config=args.variants==3 and args.seed==3407 and args.row_cap==500
        if same_config:
            for label in predictions:
                if summary["models"][label]!=frozen["models"][label]:
                    raise ValueError("Diagnostic replay differs from recorded model summary: "+label)
            if summary["paired"]!=frozen["paired"]:raise ValueError("Paired diagnostic summary changed")
        summary["reference_analysis_verified"]=same_config
    for filename,value in [("summary.json",summary),("case-analysis.json",details),("numeric-variants.json",variants_meta)]:
        (output/filename).write_text(json.dumps(value,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/reproduced-t4-run-001"))
    sources=parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--archive",type=Path)
    sources.add_argument("--database-dir",type=Path)
    sources.add_argument("--download-fixtures",action="store_true")
    parser.add_argument("--variants",type=int,default=3)
    parser.add_argument("--seed",type=int,default=3407)
    parser.add_argument("--row-cap",type=int,default=500)
    run(parser.parse_args())

if __name__=="__main__":main()
