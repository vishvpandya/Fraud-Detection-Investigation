# Databricks notebook source
silver_txn = spark.table("workspace.graph_fraud.silver_transactions")
display(silver_txn.limit(5))


# COMMAND ----------

from pyspark.sql.functions import (
    count, countDistinct, sum, min, max
)

account_features = silver_txn.groupBy("account_id").agg(
    count("*").alias("txn_count"),
    sum("amount").alias("total_amount"),
    countDistinct("device_id").alias("unique_devices"),
    countDistinct("ip_address").alias("unique_ips"),
    min("event_ts").alias("first_txn_ts"),
    max("event_ts").alias("last_txn_ts")
)

display(account_features.limit(5))


# COMMAND ----------

from pyspark.sql.functions import datediff, col

account_features = account_features.withColumn(
    "account_active_days",
    datediff(col("last_txn_ts"), col("first_txn_ts")) + 1
)

display(account_features.limit(5))

# COMMAND ----------

if_df = account_features.select(
    "account_id",
    "txn_count",
    "total_amount",
    "unique_devices",
    "unique_ips",
    "account_active_days"
)


# COMMAND ----------

if_pd = if_df.toPandas().fillna(0)


# COMMAND ----------

from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # assume ~5% suspicious
    random_state=42
)

if_pd["anomaly_flag"] = iso.fit_predict(
    if_pd.drop(columns=["account_id"])
)

# Convert to binary anomaly
if_pd["is_anomaly"] = (if_pd["anomaly_flag"] == -1).astype(int)


# COMMAND ----------

silver_account_anomalies = spark.createDataFrame(
    if_pd[["account_id", "is_anomaly"]]
)


# COMMAND ----------

silver_account_anomalies.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.graph_fraud.silver_account_anomalies")

print("✅ Silver account anomaly table created")


# COMMAND ----------

spark.sql("""
SELECT is_anomaly, COUNT(*) AS cnt
FROM workspace.graph_fraud.silver_account_anomalies
GROUP BY is_anomaly
""").show()


# COMMAND ----------

# Join features + anomaly flag
viz_df = account_features.join(
    spark.table("workspace.graph_fraud.silver_account_anomalies"),
    on="account_id",
    how="inner"
)

# Convert small sample to Pandas for plotting
viz_pd = viz_df.select(
    "txn_count",
    "total_amount",
    "is_anomaly"
).sample(fraction=1.0, seed=42).toPandas()


# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Normal accounts
normal = viz_pd[viz_pd["is_anomaly"] == 0]
plt.scatter(
    normal["txn_count"],
    normal["total_amount"],
    c="blue",
    alpha=0.5,
    label="Normal"
)

# Fraud / anomaly accounts
fraud = viz_pd[viz_pd["is_anomaly"] == 1]
plt.scatter(
    fraud["txn_count"],
    fraud["total_amount"],
    c="red",
    alpha=0.9,
    label="Fraud (Anomaly)"
)

plt.xlabel("Transaction Count")
plt.ylabel("Total Transaction Amount")
plt.title("Isolation Forest – Fraud vs Normal Accounts")
plt.legend()
plt.grid(True)

plt.show()


# COMMAND ----------

feature_cols = [
    "txn_count",
    "total_amount",
    "unique_devices",
    "unique_ips",
    "account_active_days"
]

viz_df = account_features.join(
    spark.table("workspace.graph_fraud.silver_account_anomalies"),
    on="account_id",
    how="inner"
)

viz_pd = viz_df.select(feature_cols + ["is_anomaly"]) \
               .toPandas() \
               .fillna(0)


# COMMAND ----------

from sklearn.preprocessing import StandardScaler

X = viz_pd[feature_cols].values
y = viz_pd["is_anomaly"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# COMMAND ----------

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

viz_pd["x"] = X_2d[:, 0]
viz_pd["y"] = X_2d[:, 1]


# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

# Normal accounts (blue clusters)
plt.scatter(
    viz_pd[viz_pd["is_anomaly"] == 0]["x"],
    viz_pd[viz_pd["is_anomaly"] == 0]["y"],
    c="steelblue",
    alpha=0.6,
    s=30,
    label="Normal Accounts"
)

# Fraud / anomaly accounts (red dots)
plt.scatter(
    viz_pd[viz_pd["is_anomaly"] == 1]["x"],
    viz_pd[viz_pd["is_anomaly"] == 1]["y"],
    c="red",
    alpha=0.9,
    s=60,
    edgecolors="black",
    label="Fraud / Anomalies"
)

plt.title("Account Behavior Clusters (Isolation Forest + PCA)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend()
plt.grid(True)

plt.show()


# COMMAND ----------

