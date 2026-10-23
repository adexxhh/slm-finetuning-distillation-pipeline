import json,sqlite3
from argparse import Namespace
from experiments.spider import execute,schema_from_db,score
from evals.benchmark import record_id

def test_populated_execution_matching_and_readonly(tmp_path):
    db=tmp_path/"fixture.sqlite"
    with sqlite3.connect(db) as c:
        c.executescript("CREATE TABLE items(id INTEGER PRIMARY KEY, value TEXT); INSERT INTO items VALUES(1,'a'),(2,'b');")
    schema,prompt=schema_from_db(db)
    assert "items" in schema and "value" in prompt
    ok,rows,ordered,error=execute(db,"SELECT value FROM items ORDER BY id")
    assert ok and rows==[("a",),("b",)] and ordered
    assert not execute(db,"DELETE FROM items")[0]
    record={"db_id":"fixture","database":str(db),"conversations":[
        {"from":"system","value":prompt},{"from":"human","value":"List values"},
        {"from":"gpt","value":"SELECT value FROM items ORDER BY id"}]}
    test=tmp_path/"test.jsonl";test.write_text(json.dumps(record)+"\n")
    pred=tmp_path/"pred.jsonl";out=tmp_path/"score.json"
    pred.write_text(json.dumps({"id":record_id(record),"sql":"SELECT value FROM items ORDER BY id DESC"})+"\n")
    score(Namespace(test_file=test,predictions=pred,output=out))
    assert json.loads(out.read_text())["correct"]==0
    pred.write_text(json.dumps({"id":record_id(record),"sql":"SELECT value FROM items ORDER BY id"})+"\n")
    score(Namespace(test_file=test,predictions=pred,output=out))
    report=json.loads(out.read_text())
    assert report["correct"]==1 and report["nonempty_reference_count"]==1

def test_result_budget(tmp_path):
    db=tmp_path/"fixture.sqlite"
    with sqlite3.connect(db) as c:c.execute("CREATE TABLE t(x)")
    ok,_,_,error=execute(db,"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<10002) SELECT x FROM n")
    assert not ok and "budget" in error
