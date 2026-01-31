# Databricks notebook source
# Gold graph nodes (accounts, devices, IPs)
nodes_df = spark.table("workspace.graph_fraud.gold_graph_nodes")

# Gold graph edges (graph structure only)
edges_df = spark.table("workspace.graph_fraud.gold_graph_edges")

# Silver anomaly scores (Isolation Forest output)
anomaly_df = spark.table("workspace.graph_fraud.silver_account_anomalies")

# Silver transactions (real transaction attributes)
silver_txn = spark.table("workspace.graph_fraud.silver_transactions")

display(nodes_df.limit(5))
display(edges_df.limit(5))
display(anomaly_df.limit(5))
display(silver_txn.limit(5))



# COMMAND ----------

account_nodes = nodes_df.filter(nodes_df.node_type == "account")

# IMPORTANT: drop placeholder anomaly column if present
account_nodes = account_nodes.drop("is_anomaly")

display(account_nodes.limit(5))


# COMMAND ----------

edges_enriched = edges_df.join(
    silver_txn,
    edges_df.src == silver_txn['account_id'],
    "left"
)

display(edges_enriched.limit(5))


# COMMAND ----------

from pyspark.sql.functions import sum, avg, countDistinct

txn_features = edges_enriched.groupBy("src").agg(
    sum("Amount").alias("total_amount"),
    avg("Amount").alias("avg_amount"),
    countDistinct("dst").alias("unique_targets")
).withColumnRenamed("src", "node_id")

display(txn_features.limit(5))


# COMMAND ----------

anomaly_df_fixed = anomaly_df.withColumnRenamed(
    "account_id", "node_id"
)

display(anomaly_df_fixed.limit(5))


# COMMAND ----------

eval_df = account_nodes \
    .join(txn_features, on="node_id", how="left") \
    .join(anomaly_df_fixed, on="node_id", how="left") \
    .fillna(0)

display(eval_df.limit(5))



# COMMAND ----------

from pyspark.sql.functions import rand

eval_df = eval_df.withColumn(
    "fraud_probability",
    0.6 * eval_df.is_anomaly + 0.4 * rand()
)

display(eval_df.select("node_id", "fraud_probability", "is_anomaly").limit(5))


# COMMAND ----------

from pyspark.sql.functions import when, col

FRAUD_THRESHOLD = 0.16087716380862072

scored_df = eval_df.withColumn(
    "fraud_status",
    when(
        (col("fraud_probability") >= FRAUD_THRESHOLD) | (col("is_anomaly") == 1),
        "FRAUD"
    ).otherwise("NORMAL")
)

scored_df.groupBy("fraud_status").count().show()


scored_df = scored_df.withColumn(
    "risk_level",
    when(col("fraud_probability") >= 0.8, "HIGH_RISK")
    .when(col("fraud_probability") >= 0.4, "MEDIUM_RISK")
    .when(col("fraud_probability") >= 0.2, "LOW_RISK")
    .otherwise("NORMAL")
)

suspicious_accounts = scored_df.filter(
    col("risk_level").isin("HIGH_RISK", "MEDIUM_RISK")
).select(
    col("node_id").alias("account_id"),
    "risk_level",
    "fraud_probability"
)


# COMMAND ----------

fraud_accounts = scored_df.filter(col("fraud_status") == "FRAUD")

display(
    fraud_accounts.select(
        "node_id",
        "fraud_probability",
        "is_anomaly",
        "total_amount",
        "avg_amount",
        "unique_targets"
    ).orderBy(col("fraud_probability").desc())
)


# COMMAND ----------

from pyspark.sql.functions import percentile_approx

amount_threshold = scored_df.select(
    percentile_approx("total_amount", 0.90)
).collect()[0][0]

eval_labeled = scored_df.withColumn(
    "synthetic_label",
    when(
        (col("total_amount") > amount_threshold) |
        (col("unique_targets") > 5),
        1
    ).otherwise(0)
)

display(eval_labeled.select("node_id", "synthetic_label").limit(5))


# COMMAND ----------

eval_labeled = eval_labeled.withColumn(
    "prediction",
    when(col("fraud_status") == "FRAUD", 1).otherwise(0)
)


# COMMAND ----------

import numpy as np
from sklearn.metrics import precision_recall_curve

pdf = eval_labeled.select("synthetic_label", "prediction", "fraud_probability").toPandas()

y_true = pdf["synthetic_label"]
y_scores = pdf["fraud_probability"]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

beta = 2  # bias toward recall

f_beta = (1 + beta**2) * (precision * recall) / (
    beta**2 * precision + recall + 1e-9
)

best_idx = np.argmax(f_beta)
best_threshold = thresholds[best_idx]

print("Best threshold (Recall-focused):", best_threshold)
print("Precision:", precision[best_idx])
print("Recall:", recall[best_idx])

# COMMAND ----------

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

y_true = pdf["synthetic_label"]
y_pred = pdf["prediction"]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("Accuracy:", accuracy_score(y_true, y_pred))

# COMMAND ----------

import numpy as np
from sklearn.metrics import precision_recall_curve

target_recall = 0.7

# Compute precision, recall, thresholds from pdf
precision, recall, thresholds = precision_recall_curve(pdf["synthetic_label"], pdf["fraud_probability"])

idx = np.where(recall >= target_recall)[0][-1]
threshold_recall_70 = thresholds[idx]

print("Threshold for Recall ≥ 70%:", threshold_recall_70)
print("Precision:", precision[idx])
print("Recall:", recall[idx])

# COMMAND ----------

suspicious_accounts = scored_df.filter(
    col("risk_level").isin("HIGH_RISK", "MEDIUM_RISK")
).select(
    col("node_id").alias("account_id"),
    "risk_level",
    "fraud_probability"
)

# COMMAND ----------

fraud_cases = suspicious_accounts.join(
    spark.table("workspace.graph_fraud.silver_transactions"),
    on="account_id",
    how="left"
)


# COMMAND ----------

fraud_report = fraud_cases.select(
    "account_id",
    "risk_level",
    "fraud_probability",
    "device_id",
    "ip_address",
    "transaction_id",
    "amount",
    "event_ts",
    "phone_number",
    "email"
)


# COMMAND ----------

fraud_report.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.graph_fraud.gold_fraud_cases")


# COMMAND ----------

display(
    fraud_report
    .limit(5)
)

# COMMAND ----------

display(
    fraud_report
    .orderBy(col("fraud_probability").desc())
    .limit(1000)
)


# COMMAND ----------

import pyspark.sql.functions as F

account_alerts = fraud_report.groupBy(
    "account_id", "risk_level"
).agg(
    F.count("*").alias("txn_count"),
    F.sum("amount").alias("total_amount"),
    F.max("fraud_probability").alias("max_risk"),
    F.collect_set("ip_address").alias("ips_used"),
    F.collect_set("device_id").alias("devices_used")
)

display(account_alerts.limit(10))

# COMMAND ----------

explain_df = eval_df.select(
    "node_id",
    "fraud_probability",
    "total_amount",
    "avg_amount",
    "unique_targets",
    "is_anomaly"
)


# COMMAND ----------

from pyspark.sql.functions import when, array, lit

explain_df = explain_df.withColumn(
    "fraud_reasons",
    array(
        when(col("total_amount") > 20000, lit("High total transaction volume")),
        when(col("avg_amount") > 3000, lit("Unusually large average transaction")),
        when(col("unique_targets") > 3, lit("Transfers to many accounts")),
        when(col("is_anomaly") == 1, lit("Statistically rare behavior"))
    )
)

display(explain_df.limit(100))


# COMMAND ----------

high_risk_explanations = explain_df.filter(
    col("fraud_probability") >= 0.9
)

display(high_risk_explanations.limit(10))


# COMMAND ----------

explain_features = [
    "total_amount",
    "avg_amount",
    "unique_targets",
    "is_anomaly"
]


# COMMAND ----------

display(explain_df.head(5))

# COMMAND ----------

display(explain_df.head(5))

# COMMAND ----------

account_id = "acc_7840"

row = explain_df.filter(f"node_id = '{account_id}'").collect()[0]
original_score = row["fraud_probability"]




# COMMAND ----------

baseline = explain_df.filter("is_anomaly = 0").select(
    *[F.expr(f"percentile_approx({c}, 0.5)").alias(c) for c in explain_features]
).collect()[0].asDict()

display(baseline)

# COMMAND ----------

explain_features = [
    "total_amount",
    "avg_amount",
    "unique_targets",
    "is_anomaly"
]


# COMMAND ----------

account_id = "acc_5915"

account_row = (
    explain_df
    .filter(f"node_id = '{account_id}'")
    .select(
        "fraud_probability",
        "total_amount",
        "avg_amount",
        "unique_targets",
        "is_anomaly"
    )
    .collect()[0]
)

account = account_row.asDict()
account


# COMMAND ----------

high_risk_explanations = explain_df.filter(
    col("fraud_probability") >= 0.9
)

display(high_risk_explanations.limit(10))

# COMMAND ----------


deviation = {
    "total_amount": abs(account["total_amount"] - baseline["total_amount"]),
    "avg_amount": abs(account["avg_amount"] - baseline["avg_amount"]),
    "unique_targets": abs(account["unique_targets"] - baseline["unique_targets"]),
    "is_anomaly": abs(account["is_anomaly"] - baseline["is_anomaly"])
}

deviation


# COMMAND ----------

print(deviation)


# COMMAND ----------

import builtins

total_dev = builtins.sum(deviation.values())
print("total_dev =", total_dev)


# COMMAND ----------

import builtins

total_dev = builtins.sum(deviation.values())

if total_dev == 0:
    normalized_contribution = {k: 0 for k in deviation.keys()}
else:
    normalized_contribution = {
        k: round(v / total_dev, 4)
        for k, v in deviation.items()
    }

normalized_contribution



# COMMAND ----------

fraud_prob = account["fraud_probability"]

final_contribution = {
    k: round(v * fraud_prob, 3)
    for k, v in normalized_contribution.items()
}

final_contribution


# COMMAND ----------

display(explain_df.limit(5))

# COMMAND ----------

high_risk_explanations = explain_df.filter(
    col("fraud_probability") >= 0.9
)

display(high_risk_explanations.limit(10))

# COMMAND ----------

percentage_contribution = {
    k: round(v * 100, 2)
    for k, v in normalized_contribution.items()
}

percentage_contribution


# COMMAND ----------

