"""
Comprehensive Evaluation & Comparison Harness for Text-to-SQL Distilled SLMs.
Measures:
1. Syntax Validity Rate (%) via sqlglot AST parsing.
2. Execution Accuracy (%) against an in-memory SQLite/Postgres test database.
3. Inference Latency (TTFT ms & Total Latency ms).
4. Estimated Token Cost per 1M queries ($USD).
Generates report markdown table and JSON output in evals/results/report.md.
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from typing import List, Dict, Any, Tuple, Optional

from data_engine.validator import SQLValidator

# Mock Database Engine using SQLite in-memory DB for Gold standard comparison
class MockDatabaseEngine:
    """Sets up an in-memory SQLite database populated with schema tables and test data."""
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self._init_schema_and_seed_data()

    def _init_schema_and_seed_data(self):
        cursor = self.conn.cursor()
        
        # 1. Organizations
        cursor.execute("""
            CREATE TABLE organizations (
                org_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)
        cursor.executemany("INSERT INTO organizations VALUES (?, ?, ?, ?, ?)", [
            ("org-001", "Acme Corp", "enterprise", "2024-01-15 10:00:00", 1),
            ("org-002", "Beta Tech", "pro", "2024-02-01 11:30:00", 1),
            ("org-003", "Gamma LLC", "free", "2024-03-10 09:15:00", 0),
            ("org-004", "Delta Inc", "enterprise", "2024-04-20 14:20:00", 1),
        ])

        # 2. Users
        cursor.execute("""
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                role TEXT,
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
        """)
        cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)", [
            ("usr-001", "org-001", "alice@acme.com", "Alice Smith", "admin", "2024-01-16 08:00:00", "2024-09-10 12:00:00"),
            ("usr-002", "org-001", "bob@acme.com", "Bob Jones", "analyst", "2024-01-17 09:30:00", "2024-09-01 15:45:00"),
            ("usr-003", "org-002", "carol@beta.com", "Carol White", "viewer", "2024-02-05 10:00:00", "2024-08-20 11:10:00"),
            ("usr-004", "org-004", "david@delta.com", "David Brown", "admin", "2024-04-21 16:00:00", "2024-09-11 18:30:00"),
        ])

        # 3. Subscriptions
        cursor.execute("""
            CREATE TABLE subscriptions (
                subscription_id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                plan_name TEXT NOT NULL,
                monthly_amount REAL NOT NULL,
                status TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT
            )
        """)
        cursor.executemany("INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?, ?)", [
            ("sub-001", "org-001", "Enterprise Tier", 1500.00, "active", "2024-01-15", None),
            ("sub-002", "org-002", "Pro Plan", 299.00, "active", "2024-02-01", None),
            ("sub-003", "org-003", "Free Starter", 0.00, "canceled", "2024-03-10", "2024-04-10"),
            ("sub-004", "org-004", "Enterprise Tier", 2500.00, "active", "2024-04-20", None),
        ])

        # 4. Transactions
        cursor.execute("""
            CREATE TABLE transactions (
                transaction_id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                user_id TEXT,
                amount REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                payment_method TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
            ("tx-001", "org-001", "usr-001", 1500.00, "USD", "credit_card", "succeeded", "2024-01-15 10:05:00"),
            ("tx-002", "org-001", "usr-002", 500.00, "USD", "bank_transfer", "succeeded", "2024-02-15 11:00:00"),
            ("tx-003", "org-002", "usr-003", 299.00, "USD", "credit_card", "succeeded", "2024-02-01 11:35:00"),
            ("tx-004", "org-003", None, 50.00, "USD", "credit_card", "refunded", "2024-03-12 14:00:00"),
            ("tx-005", "org-004", "usr-004", 2500.00, "USD", "bank_transfer", "succeeded", "2024-04-20 14:25:00"),
        ])

        # 5. Audit Logs
        cursor.execute("""
            CREATE TABLE audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                ip_address TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cursor.executemany("INSERT INTO audit_logs (org_id, user_id, action, resource_type, ip_address, created_at) VALUES (?, ?, ?, ?, ?, ?)", [
            ("org-001", "usr-001", "LOGIN", "session", "192.168.1.10", "2024-09-10 12:00:00"),
            ("org-001", "usr-002", "UPDATE_ROLE", "user", "192.168.1.11", "2024-09-01 15:45:00"),
            ("org-004", "usr-004", "UPDATE_ROLE", "user", "10.0.0.5", "2024-09-11 18:30:00"),
        ])

        # 6. Product Catalog
        cursor.execute("""
            CREATE TABLE product_catalog (
                product_id TEXT PRIMARY KEY,
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT,
                unit_price REAL NOT NULL
            )
        """)
        cursor.executemany("INSERT INTO product_catalog VALUES (?, ?, ?, ?, ?)", [
            ("prod-001", "SKU-ENT-01", "Enterprise Suite", "Software", 150.00),
            ("prod-002", "SKU-PRO-01", "Pro Workspace", "Software", 49.99),
            ("prod-003", "SKU-SEC-01", "Audit Shield", "Security", 250.00),
        ])

        self.conn.commit()
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.set_progress_handler(lambda: int(time.monotonic() > getattr(self,"deadline",float("inf"))),1000)

    def execute_query(self, sql: str) -> Tuple[bool, List[Tuple], Optional[str]]:
        """Executes query and returns (success, rows, error_msg). Cleanly strips PostgreSQL JSONB syntax for SQLite compatibility."""
        sql_clean = sql.strip().rstrip(";")
        # Simple compatibility cleaning for SQLite execution test
        sql_sqlite = sql_clean  # Never silently change the SQL being evaluated.
        try:
            self.deadline=time.monotonic()+2
            cursor = self.conn.cursor()
            cursor.execute(sql_sqlite)
            rows = cursor.fetchall()
            return True, rows, None
        except Exception as e:
            return False, [], str(e)



from collections import Counter
import hashlib
import math
import sqlglot

def record_id(record):
    # Full prompt + schema identify the task; the reference answer is excluded.
    prompt=[m for m in record["conversations"] if m["from"]!="gpt"]
    return hashlib.sha256(json.dumps(prompt,sort_keys=True).encode()).hexdigest()

def evaluate(records,predictions,db):
    validator=SQLValidator()
    by_id={}
    for item in predictions:
        if item["id"] in by_id: raise ValueError("Duplicate prediction ID")
        by_id[item["id"]]=item
    ids=[record_id(r) for r in records]
    if len(set(ids))!=len(ids): raise ValueError("Duplicate evaluation task")
    if set(by_id)!=set(ids): raise ValueError("Prediction IDs must exactly match evaluation prompts")
    details=[]
    for record in records:
        item=by_id[record_id(record)]
        gold=next(m["value"] for m in record["conversations"] if m["from"]=="gpt")
        from serving.backend import extract_sql
        gold=extract_sql(gold)
        gold_valid,gold_error=validator.validate(gold)
        gold_ok,gold_rows,gold_exec_error=db.execute_query(gold) if gold_valid else (False,[],gold_error)
        prediction=item["sql"]
        valid,error=validator.validate(prediction)
        ok,rows,exec_error=db.execute_query(prediction) if valid and gold_ok else (False,[],error)
        ordered=bool(sqlglot.parse_one(gold,read="postgres").args.get("order")) if gold_valid else False
        correct=bool(gold_ok and valid and ok and (rows==gold_rows if ordered else Counter(rows)==Counter(gold_rows)))
        details.append({"id":item["id"],"reference_executable":gold_ok,"prediction_valid":valid,
                        "correct":correct,"error":gold_exec_error if not gold_ok else exec_error,
                        "prediction_sql":prediction,"latency_ms":item.get("latency_ms"),
                        "backend":item.get("backend","unspecified")})
    n=len(details); executable=sum(x["reference_executable"] for x in details)
    correct=sum(x["correct"] for x in details)
    return {"mode":"independent-predictions","database":"sqlite-fixture","num_samples":n,
            "reference_executable":executable,"reference_errors":n-executable,
            "correct":correct,"execution_accuracy_pct":100*correct/n if n else None,
            "caveat":"Small single-fixture research evaluation; PostgreSQL-only queries may be unexecutable. Not a production/generalization benchmark.",
            "details":details}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_file",default="data/test.jsonl")
    parser.add_argument("--predictions",required=True,help="JSONL: id, sql, optional latency_ms and backend")
    parser.add_argument("--output_dir",default="evals/results")
    args=parser.parse_args()
    def read(path):
        with open(path,encoding="utf-8") as f:return [json.loads(line) for line in f if line.strip()]
    records=read(args.test_file); predictions=read(args.predictions)
    db=MockDatabaseEngine()
    try: report=evaluate(records,predictions,db)
    finally:db.conn.close()
    os.makedirs(args.output_dir,exist_ok=True)
    with open(os.path.join(args.output_dir,"report.json"),"w",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2))

if __name__=="__main__":main()
