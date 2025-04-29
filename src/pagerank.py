import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, lit, collect_list, desc, size
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from typing import Optional
from ascii import printConverged


def compute_pagerank2(
    spark: SparkSession, input_path: str, output_path: str, tolerance: float = 1e-4
):  # Load JSON data
    df = spark.read.option("multiLine", True).json(input_path)

    # Create edges DataFrame (src, dst)
    edges = df.select(
        col("id").alias("src"), explode(col("references")).alias("dst")
    ).cache()
    edges.count()

    # List of all unique nodes
    nodes = (
        df.select(col("id"))
        .union(edges.select(col("dst").alias("id")))
        .distinct()
        .cache()
    )

    num_nodes = nodes.count()

    # Initialize ranks: all nodes start with equal rank summing to 1
    ranks = nodes.withColumn("rank", lit(1.0 / num_nodes)).cache()
    ranks.count()

    # Count outgoing links
    out_degree = edges.groupBy("src").count().cache()
    out_degree.count()

    # Iterative PageRank parameters
    num_iterations = 50
    damping_factor = 0.85

    tmpdir = "spark-tmp"
    dead_ends = nodes.join(out_degree, nodes.id == out_degree.src, "left_anti")

    ranks.write.mode("overwrite").parquet(tmpdir)
    for i in range(num_iterations):
        # Join edges with ranks
        print("Iteration", i + 1)
        ranks = spark.read.parquet(tmpdir)
        joined = edges.join(ranks, edges.src == ranks.id, "left").select(
            edges.src, edges.dst, ranks.rank
        )

        # Join with out_degree to calculate contribution
        joined = joined.join(out_degree, "src", "left")

        # Each node contributes rank / out_degree to each of its outgoing neighbors
        contributions = joined.withColumn(
            "contribution", col("rank") / col("count")
        ).select("dst", "contribution")

        # Sum of contributions received by each node
        contribs_per_node = contributions.groupBy("dst").agg(
            F.sum("contribution").alias("sum_contribution")
        )

        # Handle dead ends: total rank lost due to nodes with no outgoing edges
        dead_end_rank = (
            ranks.join(dead_ends, "id", "inner")
            .agg(F.sum("rank").alias("dead_end_rank"))
            .collect()[0]["dead_end_rank"]
        )

        if dead_end_rank is None:
            dead_end_rank = 0.0

        # Distribute dead_end_rank evenly across all nodes
        dead_end_contrib = dead_end_rank / num_nodes

        ranks = (
            nodes.join(contribs_per_node, nodes.id == contribs_per_node.dst, "left")
            .withColumn(
                "sum_contribution", F.coalesce(col("sum_contribution"), F.lit(0.0))
            )
            .select(
                nodes.id,
                (
                    (1.0 - damping_factor) / num_nodes
                    + damping_factor * (col("sum_contribution") + dead_end_contrib)
                ).alias("rank"),
            )
        )

        previous_ranks = spark.read.parquet(tmpdir)
        total_rank = ranks.agg(F.sum("rank").alias("total")).collect()[0]["total"]
        err = (
            ranks.join(previous_ranks, "id")
            .select(F.abs(ranks.rank - previous_ranks.rank).alias("diff"))
            .agg(F.sum("diff"))
            .collect()[0][0]
        )
        print("\tError:", err)
        print("\tTotal rank:", total_rank)
        ranks.write.mode("overwrite").parquet(tmpdir)
        if err < tolerance:
            print("\tALGORITHM CONVERGED IN", i + 1, "ITERATIONS")
            printConverged()
            break

    ranks = spark.read.parquet(tmpdir)
    total_rank = ranks.agg(F.sum("rank").alias("total")).collect()[0]["total"]
    ranks = ranks.withColumn("rank", col("rank") / total_rank)
    ranks.write.mode("overwrite").parquet(output_path)


def get_top_n_ranked_nodes(pagerank_parquet_path, n):
    # Written by ChatGPT
    spark = (
        SparkSession.builder.config("spark.sql.caseSensitive", "true")
        .appName("TopPageRankRetriever")
        .getOrCreate()
    )

    pr_df = spark.read.parquet(pagerank_parquet_path)

    top_n = (
        pr_df.orderBy(desc("rank"))
        .limit(n)
        .select("id", "rank")
        .rdd.map(lambda row: (row["id"], row["rank"]))
        .collect()
    )

    return top_n


def sum_all_pageranks(pagerank_parquet_path: str) -> float:
    spark = (
        SparkSession.builder.config("spark.sql.caseSensitive", "true")
        .appName("TopPageRankRetriever")
        .getOrCreate()
    )
    df = spark.read.parquet(pagerank_parquet_path)
    df.show(5)
    result = df.select(F.sum(col("rank"))).collect()

    result = result if result is not None else 0
    return result
