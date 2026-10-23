"""Read-only AST and schema checks. Database permissions remain the final boundary."""
from typing import Optional
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from data_engine.schemas import ENTERPRISE_SCHEMAS

class SQLValidationError(ValueError): pass

class SQLValidator:
    def __init__(self, schema_definitions=None):
        self.schemas = schema_definitions if schema_definitions is not None else ENTERPRISE_SCHEMAS

    def validate(self, sql):
        if not isinstance(sql, str) or not sql.strip():
            return False, "Empty SQL query"
        try:
            statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
            if len(statements) != 1:
                raise SQLValidationError("Exactly one statement is required")
            root = statements[0]
            if not isinstance(root, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
                raise SQLValidationError("Only SELECT queries are allowed")
            banned = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
                      exp.TruncateTable, exp.Command, exp.Merge, exp.Grant, exp.Into, exp.Lock)
            if any(isinstance(n, banned) for n in root.walk()):
                raise SQLValidationError("Write, DDL, and locking clauses are forbidden")
            # Conservative function policy for model-generated SQL.
            safe = {"abs","avg","cast","ceil","ceiling","coalesce","concat","count","date",
                    "date_trunc","extract","floor","greatest","least","length","lower","max",
                    "min","nullif","round","row_number","sum","upper","lag","lead","rank",
                    "dense_rank","substring","trim","current_date","current_timestamp","and","or",
                    "timestamp_trunc","exists","json_extract","json_extract_scalar","j_s_o_n_b_contains_top_key","anonymous"}
            for func in root.find_all(exp.Func):
                name = func.name.lower() if isinstance(func,exp.Anonymous) else func.sql_name().lower()
                if name not in safe - {"anonymous"}:
                    raise SQLValidationError(f"Function '{name}' is not allowlisted")
            ctes = {c.alias.lower() for c in root.find_all(exp.CTE)}
            for table in root.find_all(exp.Table):
                if table.catalog or table.db:
                    raise SQLValidationError("Qualified external schemas are not allowed")
                if table.name.lower() not in ctes and table.name.lower() not in self.schemas:
                    raise SQLValidationError(f"Unknown table '{table.name}'")
            schema = {table: {col: "UNKNOWN" for col in spec["columns"]} for table,spec in self.schemas.items()}
            qualify(root.copy(), dialect="postgres", schema=schema, validate_qualify_columns=True,
                    identify=False, quote_identifiers=False)
            return True, None
        except Exception as exc:
            return False, str(exc)

def validate_sql_query(sql):
    return SQLValidator().validate(sql)
