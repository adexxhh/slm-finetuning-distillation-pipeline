"""Generate independent model predictions; reference answers never enter inference."""
import argparse,json,time
from serving.backend import HFBackend
from evals.benchmark import record_id
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--adapter")
    p.add_argument("--test_file",default="data/test.jsonl")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    backend=HFBackend(args.model,args.adapter)
    with open(args.test_file,encoding="utf-8") as f,open(args.output,"w",encoding="utf-8") as out:
        for line in f:
            if not line.strip():continue
            record=json.loads(line)
            schema=next(m["value"] for m in record["conversations"] if m["from"]=="system")
            question=next(m["value"] for m in record["conversations"] if m["from"]=="human")
            start=time.perf_counter()
            result=backend.generate(question,schema)
            out.write(json.dumps({"id":record_id(record),"sql":result["sql"],
                       "latency_ms":(time.perf_counter()-start)*1000,"backend":result["backend"]})+"\n")
if __name__=="__main__":main()
