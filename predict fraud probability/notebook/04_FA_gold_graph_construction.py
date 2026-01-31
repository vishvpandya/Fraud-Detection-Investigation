# Databricks notebook source
silver_txn = spark.table("workspace.graph_fraud.silver_transactions")
silver_anomaly = spark.table("workspace.graph_fraud.silver_account_anomalies")

display(silver_txn.limit(5))


# COMMAND ----------

from pyspark.sql.functions import lit

account_nodes = silver_txn.select("account_id") \
    .distinct() \
    .join(silver_anomaly, on="account_id", how="left") \
    .fillna({"is_anomaly": 0}) \
    .withColumn("node_type", lit("account"))

display(account_nodes.limit(5))


# COMMAND ----------

device_nodes = silver_txn.select("device_id") \
    .distinct() \
    .withColumnRenamed("device_id", "node_id") \
    .withColumn("node_type", lit("device")) \
    .withColumn("is_anomaly", lit(0))

display(device_nodes.limit(5))


# COMMAND ----------

ip_nodes = silver_txn.select("ip_address") \
    .distinct() \
    .withColumnRenamed("ip_address", "node_id") \
    .withColumn("node_type", lit("ip")) \
    .withColumn("is_anomaly", lit(0))

display(ip_nodes.limit(5))


# COMMAND ----------

account_nodes = account_nodes \
    .withColumnRenamed("account_id", "node_id") \
    .select("node_id", "node_type", "is_anomaly")

device_nodes = device_nodes.select("node_id", "node_type", "is_anomaly")
ip_nodes = ip_nodes.select("node_id", "node_type", "is_anomaly")


# COMMAND ----------

graph_nodes = account_nodes.unionByName(device_nodes).unionByName(ip_nodes)

display(graph_nodes.limit(10))


# COMMAND ----------

graph_nodes.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.graph_fraud.gold_graph_nodes")

print("✅ Gold graph nodes created")


# COMMAND ----------

account_device_edges = silver_txn.select(
    "account_id",
    "device_id"
).distinct() \
.withColumnRenamed("account_id", "src") \
.withColumnRenamed("device_id", "dst") \
.withColumn("edge_type", lit("USES_DEVICE"))


# COMMAND ----------

account_ip_edges = silver_txn.select(
    "account_id",
    "ip_address"
).distinct() \
.withColumnRenamed("account_id", "src") \
.withColumnRenamed("ip_address", "dst") \
.withColumn("edge_type", lit("USES_IP"))


# COMMAND ----------

graph_edges = account_device_edges.unionByName(account_ip_edges)

display(graph_edges.limit(10))


# COMMAND ----------

graph_edges.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.graph_fraud.gold_graph_edges")

print("✅ Gold graph edges created")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT node_type, COUNT(*) 
# MAGIC FROM workspace.graph_fraud.gold_graph_nodes
# MAGIC GROUP BY node_type;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT edge_type, COUNT(*) 
# MAGIC FROM workspace.graph_fraud.gold_graph_edges
# MAGIC GROUP BY edge_type;
# MAGIC

# COMMAND ----------

