# Databricks notebook source
silver_df = spark.table("workspace.graph_fraud.bronze_transactions")
display(silver_df.limit(5))


# COMMAND ----------

from pyspark.sql.functions import col

silver_df = silver_df \
    .withColumnRenamed("AccountID", "account_id") \
    .withColumnRenamed("DeviceID", "device_id") \
    .withColumnRenamed("IPAddress", "ip_address") \
    .withColumnRenamed("TransactionID", "transaction_id") \
    .withColumnRenamed("Amount", "amount") \
    .withColumnRenamed("Time", "event_ts") \
    .withColumnRenamed("PhoneNumber", "phone_number") \
    .withColumnRenamed("Email", "email")

display(silver_df.limit(5))


# COMMAND ----------

from pyspark.sql.functions import lower, trim

silver_df = silver_df \
    .withColumn("account_id", trim(lower(col("account_id")))) \
    .withColumn("device_id", trim(lower(col("device_id")))) \
    .withColumn("ip_address", trim(lower(col("ip_address")))) \
    .withColumn("email", trim(lower(col("email"))))


display(silver_df.limit(5))


# COMMAND ----------

silver_df = silver_df.filter(
    col("account_id").isNotNull() &
    col("device_id").isNotNull() &
    col("ip_address").isNotNull() &
    col("amount").isNotNull() &
    col("event_ts").isNotNull()
)

display(silver_df.limit(5))

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

silver_df = silver_df.withColumn("_silver_ingest_ts", current_timestamp())

display(silver_df.limit(5))


# COMMAND ----------

silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.graph_fraud.silver_transactions")

print("✅ Silver table created: workspace.graph_fraud.silver_transactions")


# COMMAND ----------

spark.sql("""
SELECT COUNT(*) FROM workspace.graph_fraud.silver_transactions
""").show()


# COMMAND ----------

spark.sql("""
SELECT
  COUNT(DISTINCT account_id) AS accounts,
  COUNT(DISTINCT device_id) AS devices,
  COUNT(DISTINCT ip_address) AS ips
FROM workspace.graph_fraud.silver_transactions
""").show()


# COMMAND ----------

