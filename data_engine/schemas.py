"""
Enterprise Database Schemas for Text-to-SQL Distillation Data Engine.
Defines complex relational tables with multi-table joins, JSONB columns, foreign keys, and partition keys.
"""

from typing import Dict, List, Any

# Enterprise Database Schema Definitions
ENTERPRISE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "organizations": {
        "description": "Enterprise tenant organizations and subscription tiers",
        "columns": {
            "org_id": "UUID PRIMARY KEY",
            "name": "VARCHAR(255) NOT NULL",
            "tier": "VARCHAR(50) CHECK (tier IN ('free', 'pro', 'enterprise'))",
            "created_at": "TIMESTAMPTZ NOT NULL DEFAULT NOW()",
            "is_active": "BOOLEAN DEFAULT TRUE",
            "settings": "JSONB"  # e.g., {"max_users": 100, "features": ["sso", "audit"]}
        },
        "foreign_keys": {},
        "partition_key": None
    },
    "users": {
        "description": "User profiles belonging to organizations",
        "columns": {
            "user_id": "UUID PRIMARY KEY",
            "org_id": "UUID NOT NULL",
            "email": "VARCHAR(255) UNIQUE NOT NULL",
            "full_name": "VARCHAR(255)",
            "role": "VARCHAR(50) CHECK (role IN ('admin', 'analyst', 'viewer'))",
            "created_at": "TIMESTAMPTZ NOT NULL DEFAULT NOW()",
            "last_login_at": "TIMESTAMPTZ",
            "profile_meta": "JSONB"  # e.g., {"department": "Finance", "location": "US-East"}
        },
        "foreign_keys": {
            "org_id": "organizations.org_id"
        },
        "partition_key": None
    },
    "subscriptions": {
        "description": "Billing subscriptions for enterprise accounts",
        "columns": {
            "subscription_id": "UUID PRIMARY KEY",
            "org_id": "UUID NOT NULL",
            "plan_name": "VARCHAR(100) NOT NULL",
            "monthly_amount": "NUMERIC(10, 2) NOT NULL",
            "status": "VARCHAR(50) CHECK (status IN ('active', 'canceled', 'past_due'))",
            "start_date": "DATE NOT NULL",
            "end_date": "DATE"
        },
        "foreign_keys": {
            "org_id": "organizations.org_id"
        },
        "partition_key": None
    },
    "transactions": {
        "description": "Financial billing transactions partitioned by transaction date",
        "columns": {
            "transaction_id": "UUID NOT NULL",
            "org_id": "UUID NOT NULL",
            "user_id": "UUID",
            "amount": "NUMERIC(12, 2) NOT NULL",
            "currency": "VARCHAR(3) DEFAULT 'USD'",
            "payment_method": "VARCHAR(50)",
            "status": "VARCHAR(50) CHECK (status IN ('succeeded', 'failed', 'refunded'))",
            "created_at": "TIMESTAMPTZ NOT NULL",
            "metadata": "JSONB"  # e.g., {"gateway": "stripe", "ip_address": "192.168.1.1"}
        },
        "foreign_keys": {
            "org_id": "organizations.org_id",
            "user_id": "users.user_id"
        },
        "partition_key": "created_at"
    },
    "audit_logs": {
        "description": "Security and operation audit logs with JSONB payload details",
        "columns": {
            "log_id": "BIGSERIAL PRIMARY KEY",
            "org_id": "UUID NOT NULL",
            "user_id": "UUID",
            "action": "VARCHAR(100) NOT NULL",
            "resource_type": "VARCHAR(100)",
            "ip_address": "INET",
            "created_at": "TIMESTAMPTZ NOT NULL DEFAULT NOW()",
            "details": "JSONB"  # e.g., {"changed_fields": ["role"], "old_val": "viewer", "new_val": "admin"}
        },
        "foreign_keys": {
            "org_id": "organizations.org_id",
            "user_id": "users.user_id"
        },
        "partition_key": "created_at"
    },
    "product_catalog": {
        "description": "Products offered across enterprise tiers",
        "columns": {
            "product_id": "UUID PRIMARY KEY",
            "sku": "VARCHAR(100) UNIQUE NOT NULL",
            "name": "VARCHAR(255) NOT NULL",
            "category": "VARCHAR(100)",
            "unit_price": "NUMERIC(10, 2) NOT NULL",
            "attributes": "JSONB"  # e.g., {"license_type": "per_seat", "max_seats": 50}
        },
        "foreign_keys": {},
        "partition_key": None
    }
}


def get_schema_context_prompt() -> str:
    """Generates a detailed prompt representation of the database schema for LLM context."""
    lines = ["PostgreSQL Database Schema Context:\n"]
    for table_name, meta in ENTERPRISE_SCHEMAS.items():
        lines.append(f"Table: {table_name}")
        lines.append(f"  Description: {meta['description']}")
        if meta.get("partition_key"):
            lines.append(f"  Partition Key: {meta['partition_key']}")
        lines.append("  Columns:")
        for col_name, col_type in meta["columns"].items():
            lines.append(f"    - {col_name} ({col_type})")
        if meta["foreign_keys"]:
            lines.append("  Foreign Keys:")
            for fk_col, ref in meta["foreign_keys"].items():
                lines.append(f"    - {fk_col} -> {ref}")
        lines.append("")
    return "\n".join(lines)


def get_all_table_names() -> List[str]:
    """Returns list of valid table names defined in the enterprise schema."""
    return list(ENTERPRISE_SCHEMAS.keys())


def get_table_columns(table_name: str) -> List[str]:
    """Returns list of valid column names for a given table."""
    if table_name in ENTERPRISE_SCHEMAS:
        return list(ENTERPRISE_SCHEMAS[table_name]["columns"].keys())
    return []
