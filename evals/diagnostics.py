"""Result comparison and reproducible database-variation diagnostics.

Column permutation and numeric perturbations are sensitivity checks, not official
Spider metrics or proofs of semantic equivalence.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib,itertools,json,random,sqlite3,time
import sqlglot
from sqlglot import exp

@dataclass
class QueryResult:
    rows: list
    width: int
    ordered: bool
    error: str | None = None

def run_query(connection, sql, timeout=2, row_budget=10000):
    try:
        statements=[s for s in sqlglot.parse(sql,read="sqlite") if s is not None]
        if len(statements)!=1 or not isinstance(statements[0],(exp.Select,exp.Union,exp.Intersect,exp.Except)):
            raise ValueError("Exactly one read-only SELECT expression is required")
        end=time.monotonic()+timeout
        connection.set_progress_handler(lambda:int(time.monotonic()>end),1000)
        cursor=connection.execute(sql)
        rows=cursor.fetchmany(row_budget+1)
        if len(rows)>row_budget:raise ValueError("Evaluation row budget exceeded")
        return QueryResult(rows,len(cursor.description or ()),bool(statements[0].args.get("order")))
    except Exception as exc:
        return QueryResult([],0,False,str(exc))
    finally:
        connection.set_progress_handler(None,0)

def compare_results(reference, prediction, max_permutation_columns=4):
    if reference.error or prediction.error:
        return {"strict":False,"column_permutation_only":False,"mapping":None}
    def same(a,b):
        return a==b if reference.ordered else Counter(a)==Counter(b)
    strict=reference.width==prediction.width and same(reference.rows,prediction.rows)
    if strict:return {"strict":True,"column_permutation_only":False,"mapping":None}
    if (reference.width!=prediction.width or not reference.rows or
            not 1<reference.width<=max_permutation_columns):
        return {"strict":False,"column_permutation_only":False,"mapping":None}
    # One consistent column permutation must work for the whole result.
    for order in itertools.permutations(range(reference.width)):
        if same(reference.rows,[tuple(row[i] for i in order) for row in prediction.rows]):
            return {"strict":False,"column_permutation_only":True,"mapping":list(order)}
    return {"strict":False,"column_permutation_only":False,"mapping":None}

def quote_identifier(name):
    return '"'+name.replace('"','""')+'"'

def numeric_variant(source, seed, row_cap=500):
    """Copy a database in memory and shuffle non-key numeric values within columns.

    Row identities/counts, NULL positions and declared key values are preserved.
    Updates violating CHECK/other constraints are rolled back per column.
    This changes numeric correlations, not the schema or textual values.
    """
    if row_cap<2:raise ValueError("row_cap must be at least two")
    target=sqlite3.connect(":memory:")
    target.deserialize(source.serialize())
    target.execute("PRAGMA query_only=OFF")
    target.execute("PRAGMA foreign_keys=ON")
    tables=[r[0] for r in target.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    excluded={table:set() for table in tables}
    for table in tables:
        q=quote_identifier(table)
        for col in target.execute("PRAGMA table_info("+q+")"):
            if col[5]:excluded[table].add(col[1].lower())
        for index in target.execute("PRAGMA index_list("+q+")"):
            if index[2]:
                excluded[table].update(r[2].lower() for r in target.execute("PRAGMA index_info("+quote_identifier(index[1])+")") if r[2])
        for fk in target.execute("PRAGMA foreign_key_list("+q+")"):
            excluded[table].add(fk[3].lower())
            if fk[2] in excluded and fk[4]:excluded[fk[2]].add(fk[4].lower())
    changes=[];skipped=[]
    triggered={r[0] for r in target.execute("SELECT DISTINCT tbl_name FROM sqlite_master WHERE type='trigger'")}
    for table in tables:
        if table in triggered:
            skipped.append({"table":table,"reason":"Table has triggers"});continue
        q=quote_identifier(table)
        columns=[r[1] for r in target.execute("PRAGMA table_info("+q+")")]
        rowkey=next((x for x in ["rowid","_rowid_","oid"] if x not in {c.lower() for c in columns}),None)
        if not rowkey:continue
        try:
            rows=target.execute("SELECT "+quote_identifier(rowkey)+", * FROM "+q+" ORDER BY "+quote_identifier(rowkey)+" LIMIT ?",(row_cap,)).fetchall()
        except sqlite3.DatabaseError:
            skipped.append({"table":table,"reason":"No accessible rowid"});continue
        if rows and not isinstance(rows[0][0],int):
            skipped.append({"table":table,"reason":"No integer rowid"});continue
        for i,column in enumerate(columns,1):
            if column.lower() in excluded[table] or column.lower().endswith("_id"):continue
            values=[(row[0],row[i]) for row in rows if row[i] is not None]
            if len(values)<2 or any(not isinstance(v,(int,float)) for _,v in values):continue
            original=[v for _,v in values]
            if len(set(original))<2:continue
            digest=hashlib.sha256(f"{seed}/{table}/{column}".encode()).digest()
            shuffled=original.copy();random.Random(int.from_bytes(digest[:8],"big")).shuffle(shuffled)
            if shuffled==original:shuffled=original[1:]+original[:1]
            target.execute("SAVEPOINT diagnostic_column")
            try:
                target.executemany("UPDATE "+q+" SET "+quote_identifier(column)+"=? WHERE "+quote_identifier(rowkey)+"=?",
                                   [(v,rowid) for (rowid,_),v in zip(values,shuffled)])
                target.execute("RELEASE diagnostic_column")
                changes.append({"table":table,"column":column,"changed_rows":sum(a!=b for a,b in zip(original,shuffled))})
            except sqlite3.DatabaseError as exc:
                target.execute("ROLLBACK TO diagnostic_column");target.execute("RELEASE diagnostic_column")
                skipped.append({"table":table,"column":column,"reason":str(exc)})
    target.commit();target.execute("PRAGMA query_only=ON")
    return target,{"seed":seed,"row_cap":row_cap,"changes":changes,"skipped":skipped,
                   "logical_changes_sha256":hashlib.sha256(json.dumps(changes,sort_keys=True).encode()).hexdigest()}

def paired_summary(details, seed=3407, draws=5000):
    """Paired counts and exploratory database-cluster bootstrap of score difference."""
    if not details:raise ValueError("Cannot summarize an empty evaluation")
    groups={}
    counts={"both_match":0,"neither_match":0,"base_only":0,"adapter_only":0}
    for d in details:
        b,a=d["base"]["strict"],d["adapter"]["strict"]
        counts["both_match" if b and a else "base_only" if b else "adapter_only" if a else "neither_match"]+=1
        groups.setdefault(d["db_id"],[]).append(int(a)-int(b))
    if draws<1:raise ValueError("draws must be positive")
    rng=random.Random(seed);keys=sorted(groups);differences=[]
    for _ in range(draws):
        sample=[v for _ in keys for v in groups[rng.choice(keys)]]
        differences.append(100*sum(sample)/len(sample))
    differences.sort()
    return {**counts,"delta_percentage_points":100*(counts["adapter_only"]-counts["base_only"])/len(details),
        "database_clusters":len(keys),"cluster_bootstrap_seed":seed,"cluster_bootstrap_draws":draws,
        "exploratory_cluster_bootstrap_95_delta_pct":[differences[int(.025*(draws-1))],differences[int(.975*(draws-1))]],
        "uncertainty_scope":"Exploratory percentile bootstrap over database clusters; small filtered sample, not population-wide significance."}
