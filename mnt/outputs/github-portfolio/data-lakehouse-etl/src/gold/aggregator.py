"""
Gold Layer: Business aggregations and star schema generation.
Produces BI-ready fact and dimension tables.
"""

from __future__ import annotations

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class GoldAggregator:
    """
    Transforms Silver tables into Gold business-ready aggregations.
    Produces: daily_sales_summary, customer_360, product_performance.
    """

    def __init__(self, spark: SparkSession, silver_base: str, gold_base: str):
        self.spark = spark
        self.silver = silver_base
        self.gold = gold_base

    def run_all(self) -> None:
        self.daily_sales_summary()
        self.customer_360()
        self.product_performance()
        print("✅ Gold aggregations complete")

    def daily_sales_summary(self) -> None:
        """Daily revenue KPIs by region and product category."""
        orders = self.spark.read.format("delta").load(f"{self.silver}/orders")
        items = self.spark.read.format("delta").load(f"{self.silver}/order_items")
        products = self.spark.read.format("delta").load(f"{self.silver}/products")

        df = (
            orders
            .filter(F.col("is_deleted") == False)
            .join(items, "order_id", "left")
            .join(products.select("product_id", "category", "brand"), "product_id", "left")
            .groupBy(
                F.to_date("order_date").alias("date"),
                "region",
                "category",
                "brand",
            )
            .agg(
                F.sum("revenue").alias("total_revenue"),
                F.sum("quantity").alias("total_units"),
                F.countDistinct("order_id").alias("order_count"),
                F.countDistinct("customer_id").alias("unique_customers"),
                F.avg("revenue").alias("avg_order_value"),
                F.sum(F.when(F.col("is_returned") == True, F.col("revenue")).otherwise(0))
                 .alias("return_revenue"),
            )
            .withColumn("return_rate",
                        F.col("return_revenue") / F.col("total_revenue"))
            .withColumn("_gold_updated_at", F.current_timestamp())
        )

        self._write_gold(df, "daily_sales_summary", partition_by=["date"])

    def customer_360(self) -> None:
        """Customer lifetime value and behavioral segmentation."""
        orders = self.spark.read.format("delta").load(f"{self.silver}/orders")

        window_total = Window.partitionBy("customer_id")
        window_recency = Window.partitionBy("customer_id").orderBy(F.col("order_date").desc())

        df = (
            orders
            .filter(F.col("is_deleted") == False)
            .withColumn("recency_rank", F.row_number().over(window_recency))
            .groupBy("customer_id")
            .agg(
                F.sum("revenue").alias("lifetime_value"),
                F.count("order_id").alias("total_orders"),
                F.avg("revenue").alias("avg_order_value"),
                F.min("order_date").alias("first_order_date"),
                F.max("order_date").alias("last_order_date"),
                F.datediff(F.current_date(), F.max("order_date")).alias("days_since_last_order"),
            )
            .withColumn("clv_segment",
                F.when(F.col("lifetime_value") > 10000, "VIP")
                 .when(F.col("lifetime_value") > 1000, "High Value")
                 .when(F.col("lifetime_value") > 100, "Regular")
                 .otherwise("Low Value")
            )
            .withColumn("churn_risk",
                F.when(F.col("days_since_last_order") > 180, "High")
                 .when(F.col("days_since_last_order") > 90, "Medium")
                 .otherwise("Low")
            )
            .withColumn("_gold_updated_at", F.current_timestamp())
        )

        self._write_gold(df, "customer_360")

    def product_performance(self) -> None:
        """Product-level revenue, margin, and return analysis."""
        items = self.spark.read.format("delta").load(f"{self.silver}/order_items")
        products = self.spark.read.format("delta").load(f"{self.silver}/products")

        df = (
            items
            .join(products, "product_id", "left")
            .groupBy("product_id", "product_name", "category", "brand")
            .agg(
                F.sum("revenue").alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
                F.avg("unit_price").alias("avg_selling_price"),
                F.sum(F.when(F.col("is_returned") == True, 1).otherwise(0)).alias("returns"),
            )
            .withColumn("return_rate", F.col("returns") / F.col("units_sold"))
            .withColumn("revenue_rank",
                F.rank().over(Window.partitionBy("category").orderBy(F.desc("total_revenue")))
            )
            .withColumn("_gold_updated_at", F.current_timestamp())
        )

        self._write_gold(df, "product_performance", partition_by=["category"])

    def _write_gold(self, df: DataFrame, table: str, partition_by: list = None) -> None:
        path = f"{self.gold}/{table}"
        writer = (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
        )
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.save(path)
        print(f"[Gold] ✓ {table} written to {path} ({df.count()} rows)")
