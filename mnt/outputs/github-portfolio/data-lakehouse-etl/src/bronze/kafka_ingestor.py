"""
Bronze Layer: Kafka → Delta Lake streaming ingestor.
Writes raw, immutable, append-only records with full lineage metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from src.utils.spark_session import create_spark_session


BRONZE_METADATA_COLS = [
    "_ingested_at",
    "_kafka_topic",
    "_kafka_partition",
    "_kafka_offset",
    "_source_system",
]


class KafkaBronzeIngestor:
    """
    Reads from Kafka topics and writes to Delta Lake Bronze tables.
    Preserves raw payload + full lineage metadata.
    """

    def __init__(
        self,
        spark: SparkSession,
        kafka_bootstrap: str = "localhost:9092",
        bronze_base_path: str = "/delta/bronze",
        checkpoint_base: str = "/tmp/checkpoints/bronze",
        trigger_interval: str = "30 seconds",
    ):
        self.spark = spark
        self.kafka_bootstrap = kafka_bootstrap
        self.bronze_base = bronze_base_path
        self.checkpoint_base = checkpoint_base
        self.trigger_interval = trigger_interval

    def ingest(
        self,
        topic: str,
        source_system: str = "kafka",
        starting_offsets: str = "latest",
        partition_by: Optional[list] = None,
    ) -> None:
        """
        Start a streaming ingest from a Kafka topic to Delta Bronze.

        Args:
            topic:            Kafka topic name
            source_system:    Source system label for lineage
            starting_offsets: "latest" | "earliest" | JSON offsets dict
            partition_by:     List of columns to partition the Delta table by
        """
        raw_df = self._read_kafka(topic, starting_offsets)
        enriched_df = self._add_metadata(raw_df, topic, source_system)
        self._write_delta(enriched_df, topic, partition_by or ["_ingestion_date"])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _read_kafka(self, topic: str, starting_offsets: str) -> DataFrame:
        return (
            self.spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_bootstrap)
            .option("subscribe", topic)
            .option("startingOffsets", starting_offsets)
            .option("failOnDataLoss", "false")
            .option("kafka.security.protocol", "PLAINTEXT")
            .load()
        )

    @staticmethod
    def _add_metadata(df: DataFrame, topic: str, source_system: str) -> DataFrame:
        return (
            df
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_ingestion_date", F.to_date(F.current_timestamp()))
            .withColumn("_kafka_topic", F.lit(topic))
            .withColumn("_kafka_partition", F.col("partition").cast("int"))
            .withColumn("_kafka_offset", F.col("offset").cast("long"))
            .withColumn("_source_system", F.lit(source_system))
            .withColumn("_record_id", F.expr("uuid()"))
            .withColumn("raw_payload", F.col("value").cast(StringType()))
            .drop("key", "value", "topic", "partition", "offset", "timestamp", "timestampType")
        )

    def _write_delta(self, df: DataFrame, topic: str, partition_by: list) -> None:
        table_path = f"{self.bronze_base}/{topic}"
        checkpoint_path = f"{self.checkpoint_base}/{topic}"

        query = (
            df.writeStream
            .format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_path)
            .option("mergeSchema", "true")
            .partitionBy(*partition_by)
            .trigger(processingTime=self.trigger_interval)
            .start(table_path)
        )

        print(f"[Bronze] Streaming {topic} → {table_path}")
        print(f"[Bronze] Checkpoint: {checkpoint_path}")
        query.awaitTermination()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kafka → Delta Bronze ingestor")
    p.add_argument("--topic", required=True)
    p.add_argument("--kafka", default="localhost:9092")
    p.add_argument("--bronze-path", default="/delta/bronze")
    p.add_argument("--checkpoint", default="/tmp/checkpoints/bronze")
    p.add_argument("--trigger", default="30 seconds")
    p.add_argument("--offsets", default="latest")
    args = p.parse_args()

    spark = create_spark_session(app_name=f"Bronze-{args.topic}")
    ingestor = KafkaBronzeIngestor(
        spark=spark,
        kafka_bootstrap=args.kafka,
        bronze_base_path=args.bronze_path,
        checkpoint_base=args.checkpoint,
        trigger_interval=args.trigger,
    )
    ingestor.ingest(topic=args.topic, starting_offsets=args.offsets)
