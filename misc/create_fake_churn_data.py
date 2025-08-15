import argparse, os, sys, json, math, random
from datetime import datetime, timedelta, timezone
from dateutil.parser import isoparse
from decimal import Decimal  # add at top of file

import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# ---------- Defaults you can override via CLI ----------
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")
DEFAULT_OFFLINE_BUCKET   = "ml-offline"
DEFAULT_ARTIFACTS_BUCKET = "ml-artifacts"
DEFAULT_LOGS_BUCKET      = "ml-logs"
DEFAULT_DDB_TABLE        = "features_users"

def ensure_bucket(s3_client, name, region):
    """
    Ensure S3 Bucket Exists
    """
    try:
        s3_client.head_bucket(Bucket=name)
        print(f"[ok] bucket exists: s3://{name}")
    except ClientError:
        print(f"[create] bucket: s3://{name}")
        kwargs = {"Bucket": name}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3_client.create_bucket(**kwargs)

def s3_write_parquet(df: pd.DataFrame, s3_uri: str):
    """
    pandas can write directly to s3 via fsspec if s3fs is installed;
    to keep deps minimal, write via tempfile -> upload_fileobj
    """
    import io
    bio = io.BytesIO()
    df.to_parquet(bio, index=False)
    bio.seek(0)
    assert s3_uri.startswith("s3://")
    bucket, key = s3_uri[5:].split("/", 1)
    boto3.client("s3").upload_fileobj(bio, bucket, key)

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def synthesize_users(n_users: int, seed: int = 42):
    """
    Create the Fake Data
    """
    rng = np.random.default_rng(seed)
    user_id = [f"u{100000+i}" for i in range(n_users)]
    age = rng.integers(18, 75, size=n_users)
    tenure_months = rng.integers(1, 60, size=n_users)
    is_promo_user = rng.choice([0, 1], p=[0.7, 0.3], size=n_users)
    loyalty_tier = rng.choice(["bronze","silver","gold","platinum"], p=[0.45,0.35,0.15,0.05], size=n_users)
    avg_order_value = np.round(rng.normal(60, 20, size=n_users).clip(5, 300), 2)
    df_users = pd.DataFrame({
        "user_id": user_id,
        "age": age.astype(int),
        "tenure_months": tenure_months.astype(int),
        "is_promo_user": is_promo_user.astype(int),
        "loyalty_tier": loyalty_tier,
        "avg_order_value": avg_order_value
    })
    return df_users

def make_day_features(df_users: pd.DataFrame, day: datetime, seed: int = 0):
    """
    Generate per-day features + churn label with realistic correlations.
    """
    rng = np.random.default_rng(seed)
    df = df_users.copy()
    df["ts"] = day.replace(tzinfo=timezone.utc).isoformat()

    # Some temporal noise to allow drift between days
    promo_bump = rng.normal(0, 0.15)
    # Linear logit for churn probability
    # Older age -> slightly lower churn; longer tenure -> much lower churn
    # promo users churn less; higher AOV churn less; gold/platinum churn less
    tier_map = {"bronze": 0.0, "silver": -0.2, "gold": -0.5, "platinum": -0.8}
    tier_effect = df["loyalty_tier"].map(tier_map).astype(float)

    logit = (
        0.5
        + (-0.015 * df["age"])
        + (-0.03 * df["tenure_months"])
        + (-0.7 * df["is_promo_user"])
        + (-0.004 * df["avg_order_value"])
        + tier_effect
        + promo_bump
        + rng.normal(0, 0.6, size=len(df))
    )
    p_churn = sigmoid(logit)
    df["label"] = (rng.random(len(df)) < p_churn).astype(int)

    # Keep only features your training code expects + label
    offline = df[["age","tenure_months","is_promo_user","label"]].copy()
    # You can also retain extra features if your preprocessing supports them.

    # For gateway request logs (no label)
    requests = df[["user_id","age","tenure_months","is_promo_user","ts"]].copy()

    return offline, requests

def ensure_ddb_table(table_name: str, region: str):
    """Create DynamoDB table if it doesn't exist; return a Table resource."""
    ddb = boto3.resource("dynamodb", region_name=region)
    table = ddb.Table(table_name)
    try:
        table.load()  # triggers a DescribeTable
        print(f"[ok] DynamoDB table exists: {table_name}")
        return table
    except ddb.meta.client.exceptions.ResourceNotFoundException:
        print(f"[create] DynamoDB table: {table_name}")
        table = ddb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        print(f"[ok] DynamoDB table ready: {table_name}")
        return table

def seed_ddb_features(ddb_table: str, df_users: pd.DataFrame, region: str):
    table = ensure_ddb_table(ddb_table, region)

    sample = df_users.sample(min(2000, len(df_users)), random_state=7)
    with table.batch_writer(overwrite_by_pkeys=["user_id"]) as bw:
        for _, r in sample.iterrows():
            feats = {
                "loyalty_tier": str(r["loyalty_tier"]),
                "avg_order_value": Decimal(str(r["avg_order_value"])),  # <-- Decimal, not float
            }
            bw.put_item(Item={"user_id": str(r["user_id"]), "features": feats})
    print(f"[ddb] wrote ~{len(sample)} items to {ddb_table}")


def main():
    ap = argparse.ArgumentParser(description="Seed fake churn data to S3 (+optional DDB)")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--offline-bucket", default=DEFAULT_OFFLINE_BUCKET)
    ap.add_argument("--artifacts-bucket", default=DEFAULT_ARTIFACTS_BUCKET)
    ap.add_argument("--logs-bucket", default=DEFAULT_LOGS_BUCKET)
    ap.add_argument("--ddb-table", default=DEFAULT_DDB_TABLE)
    ap.add_argument("--users", type=int, default=10000, help="number of users")
    ap.add_argument("--start-date", default=datetime.utcnow().date().isoformat(), help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--days", type=int, default=5, help="number of daily partitions to generate")
    ap.add_argument("--seed-ddb", action="store_true", help="also seed DynamoDB features_users table")
    args = ap.parse_args()

    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")

    # 1) ensure buckets
    for b in [args.offline_bucket, args.artifacts_bucket, args.logs_bucket]:
        ensure_bucket(s3, b, args.region)

    # 2) synthesize users
    df_users = synthesize_users(args.users, seed=123)

    # 3) per-day offline parquet + request logs
    start = isoparse(args.start_date).date()
    all_requests = []
    for d in range(args.days):
        day = datetime.combine(start + timedelta(days=d), datetime.min.time())
        offline_df, req_df = make_day_features(df_users, day, seed=100 + d)

        # offline features write
        dt = (start + timedelta(days=d)).isoformat()
        offline_key = f"churn/features/dt={dt}/part-0000.parquet"
        s3_uri_offline = f"s3://{args.offline_bucket}/{offline_key}"
        s3_write_parquet(offline_df, s3_uri_offline)
        print(f"[s3] wrote offline features -> {s3_uri_offline} rows={len(offline_df)}")

        # requests log write (only for last day to simulate 'recent')
        if d == args.days - 1:
            req_key = f"gateway/requests/dt={dt}/req-0000.parquet"
            s3_uri_req = f"s3://{args.logs_bucket}/{req_key}"
            s3_write_parquet(req_df, s3_uri_req)
            print(f"[s3] wrote gateway requests -> {s3_uri_req} rows={len(req_df)}")
            all_requests.append(req_df)

    # 4) baseline parquet for drift (use day 1 requests-like columns)
    baseline = pd.DataFrame({
        "age": df_users["age"],
        "tenure": df_users["tenure_months"]  # if your drift script expects 'tenure'
    })
    baseline_uri = f"s3://{args.artifacts_bucket}/churn/baseline.parquet"
    s3_write_parquet(baseline, baseline_uri)
    print(f"[s3] wrote drift baseline -> {baseline_uri} rows={len(baseline)}")

    # 5) optionally seed DynamoDB (online features)
    if args.seed_ddb:
        seed_ddb_features(args.ddb_table, df_users, args.region)

    # 6) print handy paths for your training & serving
    last_dt = (start + timedelta(days=args.days - 1)).isoformat()
    print("\n=== Ready to use ===")
    print(f"OFFLINE prefix (for training): s3://{args.offline_bucket}/churn/features/dt={last_dt}")
    print(f"BASELINE parquet (for drift):  {baseline_uri}")
    print(f"RECENT logs (for drift):       s3://{args.logs_bucket}/gateway/requests/dt={last_dt}/")
    print("DynamoDB table (online feats):", args.ddb_table)
    print("\nTip:")
    print("  python -m src.model.train \\")
    print("    --output_s3_uri s3://ml-artifacts/churn/$(date -u +%Y-%m-%dT%H-%M-%S) \\")
    print(f"    --offline_data_s3_prefix s3://{args.offline_bucket}/churn/features/dt={last_dt}")
    print()

if __name__ == "__main__":
    sys.exit(main())
    