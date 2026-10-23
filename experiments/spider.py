"""Reproducible Spider development evaluation on populated, read-only SQLite databases.
This is execution matching on one database instance, not official Spider test-suite accuracy.
"""
from pathlib import Path
from collections import Counter
import argparse,json,random,sqlite3,time,hashlib
import sqlglot
from sqlglot import exp
from data_engine.validator import SQLValidator
from serving.backend import HFBackend,extract_sql
from evals.benchmark import record_id

ATTRIBUTION={
 "dataset":"Spider (Yu et al., EMNLP 2018), CC BY-SA 4.0",
 "questions":"https://huggingface.co/datasets/xlangai/spider",
 "database_mirror":"https://huggingface.co/datasets/prem-research/spider",
 "canonical":"https://yale-lily.github.io/spider"
}

def schema_from_db(path):
    conn=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)
    try:
        schema={}
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            # Table names originate from the trusted downloaded fixture; quote identifiers.
            quoted=table.replace('"','""')
            schema[table.lower()]={"columns":{row[1].lower():row[2] or "TEXT" for row in conn.execute('PRAGMA table_info("'+quoted+'")')}}
        text="Generate exactly one read-only SQLite query, no explanation.\n"
        text+="\n".join(table+"("+", ".join(meta["columns"])+")" for table,meta in schema.items())
        return schema,text
    finally:conn.close()

def execute(path,sql,timeout=2):
    try:
        tree=sqlglot.parse_one(sql,read="sqlite")
        connection=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            end=time.monotonic()+timeout
            connection.set_progress_handler(lambda:int(time.monotonic()>end),1000)
            rows=connection.execute(sql).fetchmany(10001)
            if len(rows)>10000:return False,[],False,"Result exceeds 10,000-row evaluation budget"
            return True,rows,bool(tree.args.get("order")),None
        finally:connection.close()
    except Exception as exc:return False,[],False,str(exc)

def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download,HfApi
    from transformers import AutoTokenizer
    from training.formatting import encode_completion
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    api=HfApi();questions_sha=api.dataset_info("xlangai/spider").sha
    db_sha=api.dataset_info("prem-research/spider").sha
    dataset=load_dataset("xlangai/spider",revision=questions_sha)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    train=list(dataset["train"]);test=list(dataset["validation"])
    dbs=sorted({x["db_id"] for x in train});random.Random(3407).shuffle(dbs)
    val_dbs=set(dbs[:max(1,len(dbs)//10)])
    groups={"train":[x for x in train if x["db_id"] not in val_dbs],
            "validation":[x for x in train if x["db_id"] in val_dbs],"test":test}
    limits={"train":args.train_limit,"validation":args.val_limit,"test":args.test_limit}
    manifest={"attribution":ATTRIBUTION,"question_revision":questions_sha,"database_revision":db_sha,
              "seed":3407,"max_sequence_length":args.max_length,"splits":{}}
    for split,items in groups.items():
        random.Random(3407).shuffle(items);records=[];skipped=[]
        for item in items:
            if len(records)>=limits[split]:break
            db_id=item["db_id"]
            local=hf_hub_download("prem-research/spider",f"database/{db_id}/{db_id}.sqlite",
                                  repo_type="dataset",revision=db_sha)
            path=Path(local);schema,prompt=schema_from_db(path)
            record={"db_id":db_id,"database":str(path),"conversations":[
                {"from":"system","value":prompt},{"from":"human","value":item["question"]},
                {"from":"gpt","value":item["query"]}]}
            valid,error=SQLValidator(schema).validate(item["query"])
            ok,rows,_,exec_error=execute(path,item["query"]) if valid else (False,[],False,error)
            try:encode_completion(record,tokenizer,args.max_length)
            except ValueError as exc:ok=False;exec_error=str(exc)
            if not ok:
                skipped.append({"db_id":db_id,"question":item["question"],"reason":exec_error});continue
            records.append(record)
        if not records:raise RuntimeError("No valid records for "+split)
        (root/(split+".jsonl")).write_text("\n".join(json.dumps(r) for r in records)+"\n",encoding="utf-8")
        manifest["splits"][split]={"count":len(records),"database_ids":sorted({r["db_id"] for r in records}),"skipped":skipped}
    sets=[set(manifest["splits"][s]["database_ids"]) for s in ["train","validation","test"]]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    manifest["files_sha256"]={s:hashlib.sha256((root/(s+".jsonl")).read_bytes()).hexdigest() for s in groups}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:v["count"] for k,v in manifest["splits"].items()},indent=2))

def predict(args):
    backend=HFBackend(args.model,args.adapter)
    records=[json.loads(line) for line in Path(args.test_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    details=[]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for rec in records:
        schema=next(x["value"] for x in rec["conversations"] if x["from"]=="system")
        question=next(x["value"] for x in rec["conversations"] if x["from"]=="human")
        result=backend.generate(question,schema,max_tokens=256)
        details.append({"id":record_id(rec),"sql":result["sql"],"backend":result["backend"],
                        "latency_ms":result["generation_ms"],"total_tokens":result["total_tokens"]})
        output.write_text("\n".join(json.dumps(x) for x in details)+"\n",encoding="utf-8")
    print("Independent predictions:",len(details))

def score(args):
    records=[json.loads(x) for x in Path(args.test_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    predictions=[json.loads(x) for x in Path(args.predictions).read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id={x["id"]:x for x in predictions}
    if len(by_id)!=len(predictions) or set(by_id)!={record_id(x) for x in records}:raise ValueError("Prediction ID mismatch")
    results=[]
    for record in records:
        path=Path(record["database"]);schema,_=schema_from_db(path)
        reference=next(x["value"] for x in record["conversations"] if x["from"]=="gpt")
        gold_ok,gold_rows,ordered,gold_error=execute(path,reference)
        prediction=by_id[record_id(record)];valid,error=SQLValidator(schema).validate(prediction["sql"])
        ok,rows,_,execution_error=execute(path,prediction["sql"]) if valid else (False,[],False,error)
        match=bool(gold_ok and valid and ok and (rows==gold_rows if ordered else Counter(rows)==Counter(gold_rows)))
        results.append({"id":record_id(record),"db_id":record["db_id"],"correct":match,"empty_reference":not gold_rows,
                        "reference_executable":gold_ok,"valid":valid,"error":gold_error or execution_error,
                        "prediction":prediction["sql"],"latency_ms":prediction.get("latency_ms")})
    n=len(results);correct=sum(x["correct"] for x in results)
    nonempty=[x for x in results if not x["empty_reference"]]
    p=correct/n if n else 0;z=1.96;den=1+z*z/n if n else 1
    center=(p+z*z/(2*n))/den if n else 0
    margin=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den if n else 0
    report={"scope":"Spider public development subset, single populated database execution matching",
            "not_official_test_suite_accuracy":True,"num_samples":n,"correct":correct,
            "accuracy_pct":100*p,"wilson_95_pct":[100*(center-margin),100*(center+margin)],
            "nonempty_reference_count":len(nonempty),
            "nonempty_accuracy_pct":100*sum(x["correct"] for x in nonempty)/len(nonempty) if nonempty else None,
            "reference_errors":sum(not x["reference_executable"] for x in results),"details":results,"attribution":ATTRIBUTION}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="action",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--model",required=True);prep.add_argument("--output",default="artifacts/spider")
    prep.add_argument("--train-limit",type=int,default=1000);prep.add_argument("--val-limit",type=int,default=64)
    prep.add_argument("--test-limit",type=int,default=100);prep.add_argument("--max-length",type=int,default=2048)
    pred=sub.add_parser("predict");pred.add_argument("--model",required=True);pred.add_argument("--adapter")
    pred.add_argument("--test-file",required=True);pred.add_argument("--output",required=True)
    scoring=sub.add_parser("score");scoring.add_argument("--test-file",required=True);scoring.add_argument("--predictions",required=True);scoring.add_argument("--output",required=True)
    args=p.parse_args();{"prepare":prepare,"predict":predict,"score":score}[args.action](args)
if __name__=="__main__":main()
