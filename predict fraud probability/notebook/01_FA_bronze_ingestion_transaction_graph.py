# Databricks notebook source
dbutils.fs.ls("/Volumes/workspace/default/mynewdata/")


# COMMAND ----------

bronze_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/workspace/default/mynewdata/transaction_graph_dataset.csv")

display(bronze_df.limit(5))


# COMMAND ----------

bronze_df.printSchema()


# COMMAND ----------

spark.sql("""
CREATE SCHEMA IF NOT EXISTS workspace.graph_fraud
""")

print("✅ New schema created: workspace.graph_fraud")


# COMMAND ----------

bronze_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.graph_fraud.bronze_transactions")

print("✅ Bronze table created in workspace.graph_fraud")



# COMMAND ----------

spark.sql("""
SELECT COUNT(*) FROM workspace.graph_fraud.bronze_transactions
""").show()


# COMMAND ----------

spark.table("workspace.graph_fraud.bronze_transactions").printSchema()


# COMMAND ----------

