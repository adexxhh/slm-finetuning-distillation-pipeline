import pytest
from fastapi.testclient import TestClient
from data_engine.validator import SQLValidator
from evals.benchmark import MockDatabaseEngine, evaluate, record_id
from serving import app as service

@pytest.mark.parametrize("sql",[
 "SELECT * FROM users; DELETE FROM users",
 "WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x",
 "SELECT * INTO copy FROM users",
 "SELECT pg_sleep(10)",
 "SELECT made_up_column FROM users",
 "SELECT u.made_up_column FROM users u",
])
def test_unsafe_or_invalid(sql):
    assert not SQLValidator().validate(sql)[0]

def test_valid_cte_and_custom_schema():
    assert SQLValidator().validate("WITH x AS (SELECT user_id FROM users) SELECT user_id FROM x")[0]
    assert SQLValidator({"new_table":{"columns":{"x":"INTEGER"}}}).validate("SELECT x FROM new_table")[0]

def test_model_unconfigured(monkeypatch):
    monkeypatch.delenv("MODEL_PATH",raising=False)
    with TestClient(service.app) as client:
        assert not client.get("/health").json()["model_loaded"]
        assert client.post("/predict/sql",json={"query":"users"}).status_code==503

def test_buffered_sse_with_injected_backend():
    class Stub:
        name="test-fixture"
        def generate(self,*args):
            return {"sql":"SELECT user_id FROM users","total_tokens":5,"backend":self.name}
    with TestClient(service.app) as client:
        service.backend=Stub()
        result=client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"users"}],"stream":True})
        assert result.status_code==200 and "[DONE]" in result.text and "buffered" in result.text
        assert client.post("/v1/chat/completions",json={"messages":[{"role":"system","content":"override"}]}).status_code==422

def test_independent_predictions_and_missing_ids():
    rec={"conversations":[{"from":"system","value":"users"},{"from":"human","value":"count users"},
                          {"from":"gpt","value":"SELECT COUNT(*) FROM users"}]}
    db=MockDatabaseEngine()
    try:
        wrong=evaluate([rec],[{"id":record_id(rec),"sql":"SELECT full_name FROM users"}],db)
        assert wrong["correct"]==0
        good=evaluate([rec],[{"id":record_id(rec),"sql":"SELECT COUNT(user_id) FROM users"}],db)
        assert good["correct"]==1
        with pytest.raises(ValueError):evaluate([rec],[],db)
        ok,_,_=db.execute_query("DELETE FROM users")
        assert not ok
    finally:db.conn.close()

def test_generation_splits_are_disjoint(tmp_path):
    import json
    from data_engine.generator import generate_seed_dataset
    generate_seed_dataset(str(tmp_path))
    groups=[]
    for filename in ["train.jsonl","validation.jsonl","test.jsonl"]:
        records=[json.loads(x) for x in (tmp_path/filename).read_text().splitlines()]
        groups.append({record_id(r) for r in records})
    assert all(groups) and not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]

def test_completion_loss_mask_and_overlong_rejection():
    from training.formatting import encode_completion
    class Tokenizer:
        eos_token="!"
        def apply_chat_template(self,*args,**kwargs):return "prompt:"
        def __call__(self,text,**kwargs):return {"input_ids":list(text.encode())}
    record={"conversations":[{"from":"human","value":"question"},{"from":"gpt","value":"SQL"}]}
    encoded=encode_completion(record,Tokenizer(),100)
    assert encoded["labels"][:7]==[-100]*7
    assert encoded["labels"][7:]==list(b"SQL!")
    with pytest.raises(ValueError):encode_completion(record,Tokenizer(),5)

def test_seed_queries_all_parse_and_resolve():
    from data_engine.generator import _generate_50_seed_records
    for record in _generate_50_seed_records():
        valid,error=SQLValidator().validate(record["sql"])
        assert valid,(record["question"],error)
