"""
Configured SparkSession factory for Delta Lake + S3 + Kafka.
Includes AQE, dynamic partitioning, and optimized shuffle settings.
"""

from __future__ import annotations

import os
from typing import Optional

from pyspark.sql import SparkSession


def create_spark_session(
    app_name: str = "DataLakehouseETL",
    master: str = "local[*]",
    s3_bucket: Optional[str] = None,
    enable_hive: bool = False,
) -> SparkSession:
    """
    Create a configured SparkSession with Delta Lake, S3, and Kafka support.

    Args:
        app_name:    Spark application name
        master:      Spark master URL (local[*], yarn, spark://)
        s3_bucket:   S3 bucket for warehouse (optional)
        enable_hive: Enable Hive metastore support

    Returns:
        Configured SparkSession
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # Delta Lake
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Adaptive Query Execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        # Performance
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "200")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.parquet.mergeSchema", "true")
        # Delta Lake optimizations
        .config("spark.databricks.delta.optimizeWrite.enabled", "true")
        .config("spark.databricks.delta.autoCompact.enabled", "true")
        .config("spark.sql.delta.merge.repartitionBeforeWrite.enabled", "true")
        # Schema evolution
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    )

    if s3_bucket:
        aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        builder = (
            builder
            .config("spark.hadoop.fs.s3a.access.key", aws_key)
            .config("spark.hadoop.fs.s3a.secret.key", aws_secret)
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.sql.warehouse.dir", f"s3a://{s3_bucket}/warehouse")
        )

    if enable_hive:
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
