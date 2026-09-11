"""
Deterministic SQL Syntax, Safety, and Schema Validator.
Supports sqlglot AST parsing when installed, and provides a pure-Python regex fallback validator.
"""

import re
from typing import List, Dict, Tuple, Optional, Set

from data_engine.schemas import ENTERPRISE_SCHEMAS, get_all_table_names, get_table_columns

# Try importing sqlglot
try:
    import sqlglot
    from sqlglot import exp, parse_one, ParseError
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False


class SQLValidationError(Exception):
    """Custom exception raised when SQL fails validation."""
    pass


class SQLValidator:
    """
    Deterministic SQL syntax and schema validator.
    Enforces:
    1. PostgreSQL syntax correctness (using sqlglot or pure-Python regex engine).
    2. Read-only safety (only SELECT queries / CTE SELECT queries allowed).
    3. Schema integrity (no hallucinated tables or column names).
    """

    def __init__(self, schema_definitions: Optional[Dict[str, Dict]] = None):
        self.schemas = schema_definitions or ENTERPRISE_SCHEMAS
        self.valid_tables: Set[str] = set(self.schemas.keys())

    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates the given SQL string.
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        sql_clean = sql.strip().rstrip(";")
        if not sql_clean:
            return False, "Empty SQL query."

        if SQLGLOT_AVAILABLE:
            return self._validate_with_sqlglot(sql_clean)
        else:
            return self._validate_fallback(sql_clean)

    def _validate_with_sqlglot(self, sql: str) -> Tuple[bool, Optional[str]]:
        try:
            expression = parse_one(sql, read="postgres")
        except ParseError as e:
            return False, f"SQL Syntax Error (parse error): {str(e)}"
        except Exception as e:
            return False, f"SQL Syntax Error: {str(e)}"

        if expression is None:
            return False, "Failed to parse SQL into AST."

        # Safety Check
        forbidden_types = (
            exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
            exp.Alter, exp.TruncateTable, exp.Command, exp.Merge
        )
        if isinstance(expression, forbidden_types) or any(expression.find_all(*forbidden_types)):
            return False, "Safety Violation: Only read-only SELECT queries are allowed."

        # Schema check
        try:
            cte_aliases = {cte.alias.lower() for cte in expression.find_all(exp.CTE) if cte.alias}
            table_aliases = {}
            for table in expression.find_all(exp.Table):
                table_name = table.name.lower()
                alias = table.alias.lower() if table.alias else table_name
                if table_name in cte_aliases:
                    continue
                if table_name not in self.valid_tables:
                    return False, f"Unknown table reference '{table_name}'."
                table_aliases[alias] = table_name

            for col in expression.find_all(exp.Column):
                col_name = col.name.lower()
                if col_name == "*":
                    continue
                table_ref = col.table.lower() if col.table else None
                if table_ref and table_ref not in cte_aliases and table_ref in table_aliases:
                    actual_table = table_aliases[table_ref]
                    valid_cols = set(get_table_columns(actual_table))
                    if col_name not in valid_cols and col.parent.key not in ("jsonextract", "jsonextractscalar", "alias"):
                        return False, f"Column '{col_name}' does not exist in table '{actual_table}'."

        except Exception as e:
            return False, f"Schema Validation Error: {str(e)}"

        return True, None

    def _validate_fallback(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Pure Python fallback validator when sqlglot is not installed."""
        sql_upper = sql.upper()

        # 1. Safety check
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return False, "Safety Violation: Only SELECT or WITH queries are allowed."

        forbidden_keywords = ["INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "CREATE "]
        for kw in forbidden_keywords:
            if kw in sql_upper:
                return False, f"Safety Violation: Forbidden statement keyword '{kw.strip()}' found."

        # 2. Check table references in SQL query
        # Find FROM and JOIN clauses
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', sql.lower())

        # Collect CTE names defined in query
        cte_names = set()
        with_matches = re.findall(r'with\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as', sql.lower())
        for cte in with_matches:
            cte_names.add(cte)

        from_join_tables = re.findall(r'\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql.lower())
        for tbl in from_join_tables:
            if tbl in cte_names:
                continue
            if tbl not in self.valid_tables:
                return False, f"Unknown table reference '{tbl}' in query."

        return True, None


def validate_sql_query(sql: str) -> Tuple[bool, Optional[str]]:
    validator = SQLValidator()
    return validator.validate(sql)
