"""
Synthetic Data Generator Pipeline for Enterprise Text-to-SQL Distillation.
Supports instructor / OpenAI / local Ollama structured generation as well as an offline seed generator for immediate testing.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
from pydantic import BaseModel, Field

from data_engine.schemas import get_schema_context_prompt
from data_engine.validator import SQLValidator


class SyntheticSQLRecord(BaseModel):
    """Pydantic model for structured synthetic dataset record generation."""
    question: str = Field(description="Natural language enterprise business question or analyst query.")
    reasoning_trace: str = Field(description="Step-by-step reasoning explaining how to construct the query.")
    sql: str = Field(description="Executable read-only PostgreSQL query matching the schema.")
    complexity: str = Field(description="Query complexity level: 'simple', 'medium', or 'complex'.")


# High-quality seed queries (50 pre-generated, schema-aligned enterprise samples)
SEED_DATASET_50: List[Dict[str, str]] = [
    # 1 - Simple Join & Filtering
    {
        "question": "List all active users belonging to enterprise tier organizations along with their organization names.",
        "reasoning_trace": "To find active users in enterprise tier organizations:\n1. Join `users` and `organizations` on `org_id`.\n2. Filter for `organizations.tier = 'enterprise'` and `organizations.is_active = TRUE`.\n3. Select `users.full_name`, `users.email`, and `organizations.name`.",
        "sql": "SELECT u.full_name, u.email, o.name AS organization_name FROM users u JOIN organizations o ON u.org_id = o.org_id WHERE o.tier = 'enterprise' AND o.is_active = TRUE;",
        "complexity": "simple"
    },
    # 2 - Aggregation with HAVING
    {
        "question": "Which organizations have more than 5 users registered? Return the organization ID and user count.",
        "reasoning_trace": "Count users per organization:\n1. Group `users` table by `org_id`.\n2. Apply aggregate `COUNT(user_id)`.\n3. Filter group results using `HAVING COUNT(user_id) > 5`.",
        "sql": "SELECT org_id, COUNT(user_id) AS total_users FROM users GROUP BY org_id HAVING COUNT(user_id) > 5;",
        "complexity": "medium"
    },
    # 3 - JSONB Field Extraction
    {
        "question": "Find all users whose profile metadata department is set to 'Finance'.",
        "reasoning_trace": "Query `users` table utilizing PostgreSQL JSONB field extraction operator `->>` on `profile_meta` column for the key 'department' equal to 'Finance'.",
        "sql": "SELECT user_id, full_name, email FROM users WHERE profile_meta ->> 'department' = 'Finance';",
        "complexity": "medium"
    },
    # 4 - Date Truncation & Partitioned Aggregation
    {
        "question": "What is the total monthly transaction revenue for each currency in 2024?",
        "reasoning_trace": "1. Extract month from `created_at` using `DATE_TRUNC('month', created_at)`.\n2. Sum `amount` for succeeded transactions in 2024.\n3. Group by monthly date and currency.",
        "sql": "SELECT DATE_TRUNC('month', created_at) AS transaction_month, currency, SUM(amount) AS total_revenue FROM transactions WHERE status = 'succeeded' AND created_at >= '2024-01-01' AND created_at < '2025-01-01' GROUP BY DATE_TRUNC('month', created_at), currency ORDER BY transaction_month ASC;",
        "complexity": "complex"
    },
    # 5 - CTE & Window Functions
    {
        "question": "Rank organizations by their total successful transaction volume.",
        "reasoning_trace": "1. Use CTE to calculate sum of amounts per `org_id` in `transactions` table.\n2. Join with `organizations`.\n3. Apply `DENSE_RANK() OVER (ORDER BY total_spent DESC)`.",
        "sql": "WITH org_spending AS (SELECT org_id, SUM(amount) AS total_spent FROM transactions WHERE status = 'succeeded' GROUP BY org_id) SELECT o.name, s.total_spent, DENSE_RANK() OVER (ORDER BY s.total_spent DESC) AS revenue_rank FROM org_spending s JOIN organizations o ON s.org_id = o.org_id;",
        "complexity": "complex"
    },
    # 6 - Subquery / Exists
    {
        "question": "Find organizations that have active subscriptions but no transaction logs.",
        "reasoning_trace": "Select organizations with `status = 'active'` in subscriptions where `org_id` does NOT exist in `transactions`.",
        "sql": "SELECT o.org_id, o.name FROM organizations o JOIN subscriptions s ON o.org_id = s.org_id WHERE s.status = 'active' AND NOT EXISTS (SELECT 1 FROM transactions t WHERE t.org_id = o.org_id);",
        "complexity": "medium"
    },
    # 7 - Audit Log JSONB Analysis
    {
        "question": "Retrieve all audit log entries where the action was an update to user roles.",
        "reasoning_trace": "Filter `audit_logs` where `action = 'UPDATE_ROLE'` or `details` JSONB contains key `changed_fields`.",
        "sql": "SELECT log_id, org_id, user_id, action, created_at, details FROM audit_logs WHERE action = 'UPDATE_ROLE' OR details -> 'changed_fields' ? 'role';",
        "complexity": "medium"
    },
    # 8 - Product Catalog Filter
    {
        "question": "Find all products in the catalog with a unit price greater than $100 and license type 'per_seat'.",
        "reasoning_trace": "Query `product_catalog` filtering on `unit_price > 100` and JSONB attribute `license_type` equal to 'per_seat'.",
        "sql": "SELECT product_id, sku, name, unit_price FROM product_catalog WHERE unit_price > 100.00 AND attributes ->> 'license_type' = 'per_seat';",
        "complexity": "medium"
    },
    # 9 - Subscriptions MRR Summary
    {
        "question": "Calculate the total Monthly Recurring Revenue (MRR) for active subscriptions grouped by plan name.",
        "reasoning_trace": "Aggregate `subscriptions` table filtering for `status = 'active'`, summing `monthly_amount` per `plan_name`.",
        "sql": "SELECT plan_name, SUM(monthly_amount) AS total_mrr, COUNT(subscription_id) AS total_subscriptions FROM subscriptions WHERE status = 'active' GROUP BY plan_name ORDER BY total_mrr DESC;",
        "complexity": "simple"
    },
    # 10 - User Inactivity Query
    {
        "question": "Identify users who have never logged in or haven't logged in within the last 90 days.",
        "reasoning_trace": "Query `users` table where `last_login_at` is NULL or `last_login_at < NOW() - INTERVAL '90 days'`.",
        "sql": "SELECT user_id, org_id, email, last_login_at FROM users WHERE last_login_at IS NULL OR last_login_at < NOW() - INTERVAL '90 days';",
        "complexity": "simple"
    }
]

# Dynamically populate to 50 items with realistic variation across complex topics
def _generate_50_seed_records() -> List[Dict[str, str]]:
    dataset = list(SEED_DATASET_50)
    
    # Variations based on real-world SQL queries over the schema
    templates = [
        # 11 - 15: Audit & Security
        ("Find all audit logs created in the last 24 hours for organization '{org_id}'.",
         "Filter `audit_logs` by `org_id` and `created_at >= NOW() - INTERVAL '24 hours'.",
         "SELECT log_id, user_id, action, ip_address, created_at FROM audit_logs WHERE created_at >= NOW() - INTERVAL '24 hours';", "simple"),
        
        ("Count total security actions per user in audit logs.",
         "Group `audit_logs` by `user_id` and calculate `COUNT(log_id)`.",
         "SELECT user_id, COUNT(log_id) AS total_actions FROM audit_logs GROUP BY user_id ORDER BY total_actions DESC;", "simple"),
        
        ("Show audit logs with IP addresses coming from private subnet range '192.168.%.%",
         "Use SQL `LIKE` condition on `ip_address::text` or INET host match.",
         "SELECT log_id, org_id, action, ip_address FROM audit_logs WHERE ip_address::text LIKE '192.168.%';", "medium"),

        ("Find users who performed admin actions in audit logs.",
         "Join `audit_logs` with `users` where action starts with 'ADMIN_'.",
         "SELECT u.full_name, u.email, a.action, a.created_at FROM audit_logs a JOIN users u ON a.user_id = u.user_id WHERE a.action LIKE 'ADMIN_%';", "medium"),

        ("Get recent audit logs along with user profile metadata location.",
         "Join `audit_logs` and `users` and extract `profile_meta ->> 'location'`.",
         "SELECT a.log_id, u.email, u.profile_meta ->> 'location' AS location, a.action FROM audit_logs a JOIN users u ON a.user_id = u.user_id ORDER BY a.created_at DESC LIMIT 50;", "medium"),

        # 16 - 25: Financial & Transactions
        ("Calculate average transaction amount per payment method.",
         "Group `transactions` by `payment_method` and compute `AVG(amount)` for succeeded status.",
         "SELECT payment_method, AVG(amount) AS avg_transaction_value FROM transactions WHERE status = 'succeeded' GROUP BY payment_method;", "simple"),

        ("Find organizations with refunded transactions totaling over $500.",
         "Group `transactions` by `org_id` where `status = 'refunded'`, having `SUM(amount) > 500`.",
         "SELECT org_id, SUM(amount) AS total_refunded FROM transactions WHERE status = 'refunded' GROUP BY org_id HAVING SUM(amount) > 500.00;", "medium"),

        ("Get failed transactions with Stripe gateway details from metadata.",
         "Filter `transactions` where `status = 'failed'` and `metadata ->> 'gateway' = 'stripe'`.",
         "SELECT transaction_id, org_id, amount, created_at FROM transactions WHERE status = 'failed' AND metadata ->> 'gateway' = 'stripe';", "medium"),

        ("Calculate total transaction volume per organization tier.",
         "Join `transactions` with `organizations` on `org_id` and aggregate `SUM(amount)` grouped by `o.tier`.",
         "SELECT o.tier, SUM(t.amount) AS total_volume FROM transactions t JOIN organizations o ON t.org_id = o.org_id WHERE t.status = 'succeeded' GROUP BY o.tier;", "medium"),

        ("Find maximum single transaction amount for each currency.",
         "Group `transactions` by `currency` and return `MAX(amount)`.",
         "SELECT currency, MAX(amount) AS max_amount FROM transactions WHERE status = 'succeeded' GROUP BY currency;", "simple"),

        ("Show transaction count per user along with full user name.",
         "Join `users` and `transactions`, group by `user_id` and `full_name`.",
         "SELECT u.full_name, COUNT(t.transaction_id) AS total_tx FROM users u JOIN transactions t ON u.user_id = t.user_id GROUP BY u.user_id, u.full_name;", "medium"),

        ("List all canceled subscriptions with their start and end dates.",
         "Filter `subscriptions` where `status = 'canceled'`.",
         "SELECT subscription_id, org_id, plan_name, start_date, end_date FROM subscriptions WHERE status = 'canceled';", "simple"),

        ("Find active subscriptions where monthly amount exceeds average monthly amount.",
         "Subquery computing `AVG(monthly_amount)` for active subscriptions.",
         "SELECT subscription_id, org_id, plan_name, monthly_amount FROM subscriptions WHERE status = 'active' AND monthly_amount > (SELECT AVG(monthly_amount) FROM subscriptions WHERE status = 'active');", "complex"),

        ("Find organizations that upgraded to enterprise tier in the current year.",
         "Query `organizations` filtering by `tier = 'enterprise'` and `created_at >= '2024-01-01'`.",
         "SELECT org_id, name, created_at FROM organizations WHERE tier = 'enterprise' AND created_at >= '2024-01-01';", "simple"),

        ("Get total monthly amount spent per organization across active subscriptions.",
         "Group `subscriptions` by `org_id` and sum `monthly_amount`.",
         "SELECT o.name, SUM(s.monthly_amount) AS total_monthly_spend FROM subscriptions s JOIN organizations o ON s.org_id = o.org_id WHERE s.status = 'active' GROUP BY o.name;", "medium"),

        # 26 - 35: Users & Roles
        ("Count users per role across all organizations.",
         "Group `users` by `role`.",
         "SELECT role, COUNT(user_id) AS user_count FROM users GROUP BY role;", "simple"),

        ("List admins for inactive organizations.",
         "Join `users` and `organizations` where `role = 'admin'` and `is_active = FALSE`.",
         "SELECT u.full_name, u.email, o.name AS org_name FROM users u JOIN organizations o ON u.org_id = o.org_id WHERE u.role = 'admin' AND o.is_active = FALSE;", "medium"),

        ("Find users created in organizations with 'pro' tier.",
         "Join `users` and `organizations` filtering on `o.tier = 'pro'`.",
         "SELECT u.user_id, u.full_name, u.email FROM users u JOIN organizations o ON u.org_id = o.org_id WHERE o.tier = 'pro';", "simple"),

        ("Find organizations with no active admin users.",
         "Find organizations where `org_id` is NOT IN admin users list.",
         "SELECT org_id, name FROM organizations WHERE org_id NOT IN (SELECT org_id FROM users WHERE role = 'admin');", "complex"),

        ("Get user login statistics aggregated by month of creation.",
         "Use `DATE_TRUNC('month', created_at)` on `users` table.",
         "SELECT DATE_TRUNC('month', created_at) AS signup_month, COUNT(user_id) AS total_signups FROM users GROUP BY DATE_TRUNC('month', created_at) ORDER BY signup_month ASC;", "medium"),

        ("Find organizations whose settings allow SSO.",
         "Query `organizations` checking JSONB settings field `features` array containing 'sso'.",
         "SELECT org_id, name, tier FROM organizations WHERE settings -> 'features' ? 'sso';", "medium"),

        ("List users whose settings max_users limit in organization is over 50.",
         "Join `users` with `organizations` and extract `settings ->> 'max_users'::int`.",
         "SELECT u.user_id, u.full_name, o.name FROM users u JOIN organizations o ON u.org_id = o.org_id WHERE (o.settings ->> 'max_users')::int > 50;", "complex"),

        ("Retrieve products ordered by unit price in descending order.",
         "Select from `product_catalog` with `ORDER BY unit_price DESC`.",
         "SELECT product_id, sku, name, unit_price FROM product_catalog ORDER BY unit_price DESC;", "simple"),

        ("Count products grouped by category.",
         "Group `product_catalog` by `category`.",
         "SELECT category, COUNT(product_id) AS total_products FROM product_catalog GROUP BY category;", "simple"),

        ("Find product catalog items with max_seats attribute defined.",
         "Query `product_catalog` checking JSONB key existence `attributes ? 'max_seats'`.",
         "SELECT product_id, sku, name, attributes FROM product_catalog WHERE attributes ? 'max_seats';", "medium"),

        # 36 - 50: Advanced Multi-Table CTEs and Complex Analytics
        ("Find top 3 users by total transaction amount spent in their organization.",
         "Use CTE with `ROW_NUMBER() OVER (PARTITION BY org_id ORDER BY total_amt DESC)`.",
         "WITH ranked_users AS (SELECT user_id, org_id, amount AS total_amt, ROW_NUMBER() OVER (PARTITION BY org_id ORDER BY amount DESC) AS rnk FROM transactions WHERE status = 'succeeded') SELECT u.full_name, r.org_id, r.total_amt FROM ranked_users r JOIN users u ON r.user_id = u.user_id WHERE r.rnk <= 3;", "complex"),

        ("Calculate month-over-month growth in succeeded transactions revenue.",
         "Use CTE and `LAG()` window function over monthly aggregated transaction amounts.",
         "WITH monthly_rev AS (SELECT DATE_TRUNC('month', created_at) AS mth, SUM(amount) AS rev FROM transactions WHERE status = 'succeeded' GROUP BY DATE_TRUNC('month', created_at)) SELECT mth, rev, LAG(rev) OVER (ORDER BY mth) AS prev_rev, (rev - LAG(rev) OVER (ORDER BY mth)) AS growth FROM monthly_rev ORDER BY mth ASC;", "complex"),

        ("Find organizations with active subscriptions but total transaction volume under $1000.",
         "Join `organizations`, `subscriptions`, and `transactions` with aggregation.",
         "SELECT o.org_id, o.name, SUM(t.amount) AS total_vol FROM organizations o JOIN subscriptions s ON o.org_id = s.org_id JOIN transactions t ON o.org_id = t.org_id WHERE s.status = 'active' AND t.status = 'succeeded' GROUP BY o.org_id, o.name HAVING SUM(t.amount) < 1000.00;", "complex"),

        ("Find users with high audit log activity (>20 actions) who have never made a transaction.",
         "Use CTEs for audit log counts and transaction existence check.",
         "WITH audit_counts AS (SELECT user_id, COUNT(log_id) AS cnt FROM audit_logs GROUP BY user_id HAVING COUNT(log_id) > 20) SELECT u.user_id, u.full_name, u.email FROM users u JOIN audit_counts a ON u.user_id = a.user_id WHERE NOT EXISTS (SELECT 1 FROM transactions t WHERE t.user_id = u.user_id);", "complex"),

        ("List all transaction IDs associated with analysts role users.",
         "Join `transactions` and `users` where `role = 'analyst'`.",
         "SELECT t.transaction_id, u.email, t.amount FROM transactions t JOIN users u ON t.user_id = u.user_id WHERE u.role = 'analyst';", "medium"),

        ("Find organizations created after 2023 with 'enterprise' tier and active subscriptions.",
         "Join `organizations` and `subscriptions` with date filter.",
         "SELECT o.org_id, o.name, s.plan_name FROM organizations o JOIN subscriptions s ON o.org_id = s.org_id WHERE o.tier = 'enterprise' AND o.created_at >= '2023-01-01' AND s.status = 'active';", "medium"),

        ("Compute cumulative transaction total per organization over time.",
         "Use window function `SUM(amount) OVER (PARTITION BY org_id ORDER BY created_at)`.",
         "SELECT transaction_id, org_id, created_at, amount, SUM(amount) OVER (PARTITION BY org_id ORDER BY created_at) AS running_total FROM transactions WHERE status = 'succeeded';", "complex"),

        ("Find users whose last login was prior to their organization creation date (data anomaly check).",
         "Join `users` and `organizations` comparing timestamp fields.",
         "SELECT u.user_id, u.email, u.last_login_at, o.created_at AS org_created_at FROM users u JOIN organizations o ON u.org_id = o.org_id WHERE u.last_login_at < o.created_at;", "medium"),

        ("Get top 5 audit actions by total frequency across all organizations.",
         "Group `audit_logs` by `action` and order by count descending limit 5.",
         "SELECT action, COUNT(log_id) AS action_count FROM audit_logs GROUP BY action ORDER BY action_count DESC LIMIT 5;", "simple"),

        ("Find all products with price per seat ratio specified in attributes.",
         "Extract JSONB attributes `unit_price` and `attributes ->> 'max_seats'`.",
         "SELECT sku, name, unit_price, (unit_price / (attributes ->> 'max_seats')::numeric) AS price_per_seat FROM product_catalog WHERE attributes ? 'max_seats';", "complex"),

        ("Retrieve subscriptions expiring within the next 30 days.",
         "Filter `subscriptions` where `end_date` is between current date and current date + 30 days.",
         "SELECT subscription_id, org_id, plan_name, end_date FROM subscriptions WHERE end_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days';", "medium"),

        ("Find users having 'viewer' role who logged in during the past 7 days.",
         "Query `users` with role condition and `last_login_at` range.",
         "SELECT user_id, full_name, email FROM users WHERE role = 'viewer' AND last_login_at >= NOW() - INTERVAL '7 days';", "simple"),

        ("Calculate average number of audit logs generated per user in each organization.",
         "Nested aggregate or CTE grouping `audit_logs` by `org_id` and `user_id`.",
         "WITH user_log_counts AS (SELECT org_id, user_id, COUNT(log_id) AS num_logs FROM audit_logs GROUP BY org_id, user_id) SELECT org_id, AVG(num_logs) AS avg_logs_per_user FROM user_log_counts GROUP BY org_id;", "complex"),

        ("Identify organizations with duplicate user emails (integrity check).",
         "Group `users` by `org_id` and `email` with `HAVING COUNT(user_id) > 1`.",
         "SELECT org_id, email, COUNT(user_id) AS dup_count FROM users GROUP BY org_id, email HAVING COUNT(user_id) > 1;", "medium"),

        ("List transactions along with user department from profile metadata.",
         "Join `transactions` and `users` extracting `profile_meta ->> 'department'`.",
         "SELECT t.transaction_id, t.amount, u.email, u.profile_meta ->> 'department' AS department FROM transactions t JOIN users u ON t.user_id = u.user_id WHERE t.status = 'succeeded';", "medium")
    ]

    for q, r, s, c in templates:
        dataset.append({
            "question": q,
            "reasoning_trace": r,
            "sql": s,
            "complexity": c
        })

    return dataset[:50]


def format_sharegpt_record(system_prompt: str, question: str, reasoning_trace: str, sql: str) -> Dict[str, Any]:
    """Formats a question, reasoning trace, and SQL query into standard ShareGPT/Alpaca JSON format."""
    assistant_content = f"<thought>\n{reasoning_trace}\n</thought>\n\n```sql\n{sql}\n```"
    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": question},
            {"from": "gpt", "value": assistant_content}
        ]
    }


def generate_seed_dataset(output_dir: str = "data", train_split: float = 0.8) -> Tuple[str, str, int]:
    """
    Generates and validates 50 synthetic enterprise text-to-SQL records,
    saving them into train.jsonl and test.jsonl splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    validator = SQLValidator()
    system_prompt = get_schema_context_prompt()

    raw_records = _generate_50_seed_records()
    validated_records = []

    for item in raw_records:
        is_valid, err = validator.validate(item["sql"])
        if not is_valid:
            print(f"Warning: Seed SQL validation failed: {err}\nQuery: {item['sql']}")
            continue

        record = format_sharegpt_record(
            system_prompt=system_prompt,
            question=item["question"],
            reasoning_trace=item["reasoning_trace"],
            sql=item["sql"]
        )
        validated_records.append(record)

    split_idx = int(len(validated_records) * train_split)
    train_data = validated_records[:split_idx]
    test_data = validated_records[split_idx:]

    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for rec in train_data:
            f.write(json.dumps(rec) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test_data:
            f.write(json.dumps(rec) + "\n")

    return train_path, test_path, len(validated_records)
