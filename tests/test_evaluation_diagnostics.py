import hashlib,json,shutil,sqlite3
from pathlib import Path
import pytest
from evals.diagnostics import QueryResult,compare_results,numeric_variant,paired_summary,run_query
from experiments.reproduce import DEFAULT_EVIDENCE,materialize_databases,verified_inputs

def result(rows,width=2,ordered=False):
    return QueryResult(rows,width,ordered)

def test_column_mapping_is_global_and_does_not_change_strict_score():
    gold=result([(1,"a"),(2,"b")])
    compared=compare_results(gold,result([("b",2),("a",1)]))
    assert not compared["strict"] and compared["column_permutation_only"]
    assert compared["mapping"]==[1,0]
    assert not compare_results(result([(1,2),(3,4)]),result([(2,1),(3,4)]))["column_permutation_only"]

def test_duplicates_row_order_and_empty_projection_shape():
    assert compare_results(result([(1,2),(3,4)]),result([(3,4),(1,2)]))["strict"]
    assert not compare_results(result([(1,2),(1,2)]),result([(1,2)]))["strict"]
    assert not compare_results(result([(1,2),(3,4)],ordered=True),result([(3,4),(1,2)]))["strict"]
    assert not compare_results(result([],1),result([],2))["strict"]

def test_execution_rejects_mutation_multiple_statements_and_resource_overrun():
    c=sqlite3.connect(":memory:");c.executescript("CREATE TABLE t(x); INSERT INTO t VALUES(1),(2),(3);")
    c.execute("PRAGMA query_only=ON")
    assert run_query(c,"DELETE FROM t").error
    assert run_query(c,"SELECT 1; SELECT 2").error
    assert run_query(c,"SELECT * FROM t",row_budget=2).error
    sql="WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n"
    assert run_query(c,sql,timeout=.001).error
    assert c.execute("SELECT COUNT(*) FROM t").fetchone()==(3,)
    c.close()

def test_numeric_variants_preserve_keys_nulls_constraints_and_source():
    c=sqlite3.connect(":memory:");c.execute("PRAGMA foreign_keys=ON")
    c.executescript("""
        CREATE TABLE parent(id INTEGER PRIMARY KEY, value REAL UNIQUE);
        INSERT INTO parent VALUES(1,10),(2,20),(3,30);
        CREATE TABLE child(id INTEGER PRIMARY KEY,parent_id INTEGER REFERENCES parent(id),score REAL,lo INT,hi INT,CHECK(lo<=hi));
        INSERT INTO child VALUES(1,1,2,1,3),(2,2,8,5,7),(3,3,NULL,9,11);
        CREATE TABLE no_rowid(key TEXT PRIMARY KEY,value INT) WITHOUT ROWID;
        INSERT INTO no_rowid VALUES('a',1),('b',2);
    """)
    original=c.execute("SELECT * FROM child ORDER BY id").fetchall()
    one,meta=numeric_variant(c,3407);two,meta2=numeric_variant(c,3407)
    assert meta==meta2 and meta["changes"]
    assert one.execute("SELECT * FROM child ORDER BY id").fetchall()==two.execute("SELECT * FROM child ORDER BY id").fetchall()
    assert c.execute("SELECT * FROM child ORDER BY id").fetchall()==original
    assert one.execute("SELECT id,parent_id FROM child ORDER BY id").fetchall()==[(1,1),(2,2),(3,3)]
    assert one.execute("SELECT score IS NULL FROM child ORDER BY id").fetchall()==[(0,),(0,),(1,)]
    assert not one.execute("PRAGMA foreign_key_check").fetchall()
    assert not one.execute("SELECT * FROM child WHERE lo>hi").fetchall()
    assert one.execute("SELECT * FROM parent ORDER BY id").fetchall()==[(1,10.),(2,20.),(3,30.)]
    assert one.execute("SELECT * FROM no_rowid ORDER BY key").fetchall()==[("a",1),("b",2)]
    assert one.execute("PRAGMA query_only").fetchone()==(1,)
    one.close();two.close();c.close()

def test_numeric_probe_detects_accidental_match():
    c=sqlite3.connect(":memory:")
    c.executescript("CREATE TABLE owners(id INT PRIMARY KEY,cost INT,weight INT); INSERT INTO owners VALUES(1,100,100),(2,10,10),(3,1,1);")
    gold="SELECT id FROM owners ORDER BY cost DESC LIMIT 1"
    wrong="SELECT id FROM owners ORDER BY weight DESC LIMIT 1"
    assert compare_results(run_query(c,gold),run_query(c,wrong))["strict"]
    caught=False
    for seed in range(10):
        v,_=numeric_variant(c,seed)
        caught|=not compare_results(run_query(v,gold),run_query(v,wrong))["strict"]
        v.close()
    assert caught
    c.close()

def test_frozen_input_hashes_and_prediction_id_integrity(tmp_path):
    evidence=tmp_path/"evidence";shutil.copytree(DEFAULT_EVIDENCE,evidence)
    verified_inputs(evidence)
    target=evidence/"base-predictions.jsonl";data=target.read_text()
    target.write_text(data+"\n")
    with pytest.raises(ValueError,match="Frozen evidence hash mismatch"):verified_inputs(evidence)
    target.write_text(data.splitlines()[0]+"\n"+data)
    lock=json.loads((evidence/"evidence-lock.json").read_text())
    lock["frozen_files_sha256"]["base-predictions.jsonl"]=hashlib.sha256(target.read_bytes()).hexdigest()
    (evidence/"evidence-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError,match="Prediction ID mismatch"):verified_inputs(evidence)

def test_fixture_hash_and_path_checks(tmp_path):
    with pytest.raises(ValueError,match="Unsafe database"):
        materialize_databases({},[{"db_id":"../escape"}],tmp_path/"out",database_dir=tmp_path)
    manifest={"database_files_sha256":{"test.sqlite":"bad"}}
    (tmp_path/"test.sqlite").write_bytes(b"wrong fixture")
    with pytest.raises(ValueError,match="mismatched database"):
        materialize_databases(manifest,[{"db_id":"test"}],tmp_path/"out",database_dir=tmp_path)

def test_cluster_summary_is_paired_seeded_and_handles_empty_input():
    data=[{"db_id":db,"base":{"strict":False},"adapter":{"strict":True}} for db in ["a","a","b"]]
    one=paired_summary(data,draws=100)
    assert one==paired_summary(data,draws=100)
    assert one["adapter_only"]==3 and one["delta_percentage_points"]==100
    assert one["database_clusters"]==2 and one["exploratory_cluster_bootstrap_95_delta_pct"]==[100,100]
    with pytest.raises(ValueError):paired_summary([])


def test_score_reports_column_ambiguity_without_promoting_it_and_rejects_duplicate_ids(tmp_path):
    from argparse import Namespace
    from experiments.spider import score
    from evals.benchmark import record_id
    db=tmp_path/"db.sqlite"
    with sqlite3.connect(db) as c:c.executescript("CREATE TABLE items(id INT, name TEXT); INSERT INTO items VALUES(1,'a'),(2,'b');")
    record={"db_id":"fixture","database":str(db),"conversations":[
        {"from":"system","value":"items(id, name)"},{"from":"human","value":"List item fields"},
        {"from":"gpt","value":"SELECT id,name FROM items"}]}
    test=tmp_path/"test.jsonl";test.write_text(json.dumps(record)+"\n")
    pred=tmp_path/"pred.jsonl";pred.write_text(json.dumps({"id":record_id(record),"sql":"SELECT name,id FROM items"})+"\n")
    out=tmp_path/"score.json"
    options=Namespace(test_file=test,predictions=pred,output=out)
    score(options);report=json.loads(out.read_text())
    assert report["correct"]==0 and report["column_order_only_mismatches"]==1
    test.write_text((json.dumps(record)+"\n")*2)
    with pytest.raises(ValueError,match="Duplicate evaluation"):score(options)


def test_variant_skips_triggered_tables():
    c=sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE t(id INT PRIMARY KEY,value INT);
        INSERT INTO t VALUES(1,10),(2,20);
        CREATE TRIGGER change_key AFTER UPDATE ON t BEGIN UPDATE t SET id=id+10; END;
    """)
    v,meta=numeric_variant(c,3407)
    assert v.execute("SELECT * FROM t ORDER BY id").fetchall()==[(1,10),(2,20)]
    assert any(x["reason"]=="Table has triggers" for x in meta["skipped"])
    v.close();c.close()

def test_download_retry_and_hash_verification(tmp_path,monkeypatch):
    import io,urllib.error
    import experiments.reproduce as module
    calls=[]
    def request(url,timeout):
        calls.append(url)
        if len(calls)==1:raise urllib.error.URLError("transient")
        return io.BytesIO(b"fixture")
    monkeypatch.setattr(module.urllib.request,"urlopen",request)
    monkeypatch.setattr(module.time,"sleep",lambda _:None)
    manifest={"database_revision":"pinned","database_files_sha256":{"demo.sqlite":hashlib.sha256(b"fixture").hexdigest()}}
    paths=materialize_databases(manifest,[{"db_id":"demo"}],tmp_path/"out",download=True)
    assert paths["demo"].read_bytes()==b"fixture" and len(calls)==2
    assert all("/resolve/pinned/" in url for url in calls)
