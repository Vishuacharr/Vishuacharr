"""
Silver Layer: CDC MERGE INTO (upsert) from Bronze to Silver Delta tables.
Handles INSERT, UPDATE, DELETE CDC operations from Debezium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from delta.tables import DeltaTable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType


@dataclass
class CDCMergeConfig:
    primary_keys: List[str]
    operation_col: str = "op"         # Debezium: "c"=create, "u"=update, "d"=delete, "r"=read
    sequence_col: str = "_ingested_at"
    delete_tombstone: bool = True     # Keep deleted records with is_deleted=True
    soft_delete_col: str = "is_deleted"
    partition_cols: List[str] = field(default_factory=lambda: ["_updated_date"])


class CDCMergeProcessor:
    """
    Processes CDC events from Bronze and merges them into Silver using
    MERGE INTO (upsert) semantics.

    Supports Debezium operation codes: c/r (insert), u (update), d (delete).
    """

    def __init__(self, spark: SparkSession, config: CDCMergeConfig):
        self.spark = spark
        self.cfg = config

    def merge(
        self,
        source_df: DataFrame,
        target_path: str,
        target_schema: Optional[StructType] = None,
    ) -> None:
        """
        Merge CDC events from source into Delta target table.

        Args:
            source_df:     Incoming CDC records (from Bronze)
            target_path:   Delta table path (Silver)
            target_schema: Schema for initial table creation
        """
        # Deduplicate — keep latest per PK within each micro-batch
        deduped = self._deduplicate(source_df)

        # Initialize table if it doesn't exist
        self._init_table_if_needed(target_path, deduped, target_schema)

        target = DeltaTable.forPath(self.spark, target_path)
        join_condition = " AND ".join(
            f"target.{pk} = source.{pk}" for pk in self.cfg.primary_keys
        )

        # MERGE
        merge_builder = (
            target.alias("target")
            .merge(deduped.alias("source"), join_condition)
        )

        if self.cfg.delete_tombstone:
            merge_builder = (
                merge_builder
                .whenMatchedUpdate(
                    condition=f"source.{self.cfg.operation_col} = 'd'",
                    set={self.cfg.soft_delete_col: F.lit(True),
                         "_deleted_at": F.current_timestamp()},
                )
                .whenMatchedUpdateAll(
                    condition=f"source.{self.cfg.operation_col} IN ('u', 'r')",
                )
                .whenNotMatchedInsertAll(
                    condition=f"source.{self.cfg.operation_col} != 'd'",
                )
            )
        else:
            merge_builder = (
                merge_builder
                .whenMatchedDelete(
                    condition=f"source.{self.cfg.operation_col} = 'd'",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll(
                    condition=f"source.{self.cfg.operation_col} != 'd'",
                )
            )

        merge_builder.execute()

        # Z-ORDER on primary keys for optimal query performance
        self._optimize(target_path)

    def _deduplicate(self, df: DataFrame) -> DataFrame:
        from pyspark.sql.window import Window
        window = Window.partitionBy(*self.cfg.primary_keys).orderBy(
            F.col(self.cfg.sequence_col).desc()
        )
        return (
            df
            .withColumn("_rank", F.row_number().over(window))
            .filter(F.col("_rank") == 1)
            .drop("_rank")
            .withColumn("_updated_date", F.to_date(F.current_timestamp()))
        )

    def _init_table_if_needed(
        self,
        path: str,
        df: DataFrame,
        schema: Optional[StructType],
    ) -> None:
        try:
            DeltaTable.forPath(self.spark, path)
        except Exception:
            print(f"[Silver] Creating new Delta table at {path}")
            init_df = self.spark.createDataFrame([], schema or df.schema)
            (
                init_df.write
                .format("delta")
                .option("mergeSchema", "true")
                .partitionBy(*self.cfg.partition_cols)
                .save(path)
            )

    def _optimize(self, path: str) -> None:
        z_cols = ", ".join(self.cfg.primary_keys)
        self.spark.sql(f"OPTIMIZE delta.`{path}` ZORDER BY ({z_cols})")
